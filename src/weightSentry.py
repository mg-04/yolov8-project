#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""WeightSentry (Abumandour et al., HASP 2025) reimplemented against this harness.

The defence is two phases. Pre-deployment, compute a [min, max] bound over each
"weight unit" -- a group of weights that will share one bound. At inference,
every weight is checked against its unit's bound before use; an out-of-range
value is repaired by toggling exponent bits from MSB down until it lands back
in range, else negated, else clamped to the nearest boundary (their Algorithm 1).

Granularity is the whole design axis, and the paper's rule is architecture
dependent: per-output-channel if the kernel is larger than 3x3, otherwise fixed
groups of K weights. YOLOv8n has NO kernel larger than 3x3, so the per-channel
path never fires and everything would be K-grouped. Both are exposed here as
`mode`, because on this model per-channel is the knee of the cost curve and
K=128 costs 34x more storage for under a point of risk-weighted coverage.

Three departures from the paper, all deliberate. The first two change the
MECHANISM; `golden` changes only what the harness can measure:

  * `signed=False` keeps only the unit's maximum EXPONENT FIELD, one byte,
    instead of two fp32 bounds. That is an 8-bit unsigned compare rather than
    two float compares -- the version worth putting in an HLS datapath. It
    cannot see sign flips, which costs almost nothing: signed bounds catch
    0.50% of sign flips at per-channel granularity, 2.45% at K=128.

  * `golden` lets the caller hand over a pristine copy of the weights, so the
    harness can separate an EXACT repair (the toggle restored the original bit
    pattern) from a FALSE repair (some other toggle happened to land in range
    and was silently accepted). The paper never quantifies the second, and it
    turns out to be almost everything -- see below.

  * `two_sided=True` adds a magnitude FLOOR, one further byte per unit.

That last flag exists because Algorithm 1 as published does not restore values.
Its range test is the interval [min, max], which brackets zero, so clearing the
exponent MSB of a corrupted weight ALWAYS lands in range. Being the first
candidate tried, it is always the one accepted:

    original   corrupted    repaired        unit bound
     -2.054   -3.79e+19   -1.114e-19   [-8.735, 8.204]
      6.907    1.274e+20    3.744e-19   [-8.735, 8.204]

Measured over single flips in model.0.conv, that is 0% exact repair at every
bit except 30. The defence works in the paper's evaluation because crushing a
weight to ~1e-19 is benign, not because the weight was recovered -- it is
zero-substitution wearing a correction's clothes.

Bounding magnitude from BOTH sides fixes it. A weight's exponent must land
inside its unit's observed span, so the crush-to-zero candidate is rejected and
the loop continues to the bit that actually flipped. The bound becomes an
annulus rather than an interval, which is the right shape: the fault moves
magnitude in either direction, and an interval containing zero is blind to one
of them.
"""
import torch
import torch.nn as nn

EXP_BITS = range(30, 22, -1)          # exponent field, MSB first, per Algorithm 1


def _units(w, mode, K):
    """Group ids for every element of `w`, plus the group count.

    Returned ids index a flat (G,) bound table; the caller reshapes to w.shape.
    """
    n = w.numel()
    if mode == "chan":      # one bound for output channel
        cout = w.shape[0]
        return torch.arange(cout, device=w.device).repeat_interleave(n // cout), cout
    if mode == "tensor":    # one bound for entire layer
        return torch.zeros(n, dtype=torch.long, device=w.device), 1
    if mode == "k":         # fixed chunk k for consecutive weights in flat memory order, may be finer/coarser than `chan`
        gid = torch.arange(n, device=w.device) // K
        return gid, int(gid[-1].item()) + 1
    raise ValueError(f"mode must be chan|k|tensor, got {mode!r}")


class Bounds:
    """Pre-deployment metadata: one bound per weight unit, per conv layer."""

    def __init__(self, core, mode="chan", K=128, signed=True, two_sided=False,
                 bias_mode="chan", bias_signed=None, bias_two_sided=None):
        self.mode, self.K, self.signed, self.two_sided = mode, K, signed, two_sided
        self.bias_mode = bias_mode
        # Bound TYPE is per STREAM, not global, because the two streams fail
        # differently and cost differently:
        #
        #   weights  99.99% have |w| < 2, so a flip almost always SETS the
        #            exponent MSB and explodes. A one-sided exponent ceiling
        #            catches that at 1 B/unit; a signed range costs 8 and buys
        #            only bits that carry little damage.
        #   biases   82% have |b| < 2, so 18% of bit-30 flips COLLAPSE instead,
        #            and a bias sign flip alone costs 24.15% SDC. Worse, under
        #            "chan" grouping a bias unit holds ONE value, so lo == hi:
        #            the signed bound degenerates into an exact equality check
        #            at 4 B, catching everything and repairing exactly.
        #
        # The cheap configuration is therefore MIXED, and a single global flag
        # cannot express it. Biases default to the weight setting, so every
        # configuration built before this existed is bit-identical.
        self.bias_signed = signed if bias_signed is None else bias_signed
        self.bias_two_sided = two_sided if bias_two_sided is None else bias_two_sided
        self.lo, self.hi, self.emax, self.emin = {}, {}, {}, {}
        self.units, self.degenerate = {}, {}
        self.is_signed, self.is_two_sided = {}, {}
        for name, m in core.named_modules():
            if not isinstance(m, nn.Conv2d):
                continue
            # bias tables are keyed "<layer>#bias" so one in_range()/correct()
            # pair serves both streams without special-casing either.
            targets = [(name, m.weight.data, mode, signed, two_sided)]
            if bias_mode and m.bias is not None:
                targets.append((name + "#bias", m.bias.data, bias_mode,
                                self.bias_signed, self.bias_two_sided))
            for key, w, md, sg, ts in targets:
                self.is_signed[key], self.is_two_sided[key] = sg, ts
                gid, g = _units(w, md, K)
                flat = w.reshape(-1)
                e = ((flat.view(torch.int32) >> 23) & 0xFF).to(torch.int32)
                if sg:
                    lo = torch.full((g,), float("inf"), device=w.device)
                    hi = torch.full((g,), float("-inf"), device=w.device)
                    lo.scatter_reduce_(0, gid, flat, "amin")
                    hi.scatter_reduce_(0, gid, flat, "amax")
                    self.lo[key] = lo[gid].view_as(w)
                    self.hi[key] = hi[gid].view_as(w)
                else:
                    em = torch.zeros(g, dtype=torch.int32, device=w.device)
                    em.scatter_reduce_(0, gid, e, "amax")
                    # in_range() drops isfinite() and relies on `e <= emax`
                    # alone to reject the e == 255 of Inf/NaN, which is only
                    # sound while no unit legitimately reaches 255. Trained
                    # weights peak at 130 here; a corrupted checkpoint would
                    # not, so refuse it loudly rather than check 127x/forward.
                    if int(em.max()) >= 255:
                        raise ValueError(
                            f"{key}: unit exponent reaches 255 (Inf/NaN in the "
                            "reference weights), so a magnitude bound cannot "
                            "reject Inf/NaN faults. Fix the checkpoint.")
                    self.emax[key] = em[gid].view_as(w)
                if ts:
                    # the unit's magnitude floor. exact zeros are held out of
                    # the minimum and readmitted in in_range(), because one
                    # zero weight would otherwise drag the floor to 0 and stop
                    # the bound discriminating against a crush-to-zero repair.
                    esafe = torch.where(flat == 0, torch.full_like(e, 255), e)
                    en = torch.full((g,), 255, dtype=torch.int32, device=w.device)
                    en.scatter_reduce_(0, gid, esafe, "amin")
                    self.emin[key] = en[gid].view_as(w)
                self.units[key] = g
                # one value per unit: the bound IS that value's exponent, so
                # emin == emax and the floor is free. This is the bias case
                # under "chan" grouping, since there is one bias per channel.
                self.degenerate[key] = (g == flat.numel())

    @property
    def n_units(self):
        return sum(self.units.values())

    def _per_unit(self, key):
        """Deployed bytes for one unit's bound.

        A DEGENERATE unit holds a single value, so its bound collapses: signed
        lo == hi, which means storing the value once (4 B) -- that is plain
        duplication, not a range -- and emin == emax, so one exponent byte is
        both floor and ceiling. This is the bias case under "chan" grouping.
        """
        deg = self.degenerate[key]
        if self.is_signed[key]:
            # A degenerate unit stores its single value once (4 B) and its
            # magnitude floor is implied by that value, so two_sided is free.
            # Otherwise the floor is one more exponent byte on top of lo/hi.
            return 4 if deg else (9 if self.is_two_sided[key] else 8)
        return 1 if (deg or not self.is_two_sided[key]) else 2

    @property
    def bytes(self):
        """Metadata footprint as DEPLOYED, not as simulated: the expanded
        per-element tensors above exist only so the check vectorises."""
        return sum(g * self._per_unit(k) for k, g in self.units.items())

    @property
    def bias_bytes(self):
        return sum(g * self._per_unit(k) for k, g in self.units.items()
                   if k.endswith("#bias"))

    def in_range(self, name, x):
        """True where x is admissible. Written as a positive test so that a
        NaN or Inf payload -- every comparison against which is False -- comes
        out as NOT in range. The paper's `if w_min <= w <= w_max` has the same
        effect scalar-wise; the obvious vectorised negation (x < lo) | (x > hi)
        does not, and would let a NaN through silently.

        The exponent is extracted at most ONCE. Every elementwise op here is a
        separate CUDA kernel launch, 127 of these calls run per forward, and
        those launches -- not the memory traffic -- dominate the measured GPU
        overhead, so a redundant extraction is not free.

        An explicit isfinite() is likewise redundant in the magnitude-only
        path: Inf and NaN both carry e == 255, and __init__ refuses to build a
        bound with emax == 255, so `e <= emax` already rejects them.

        KNOWN BLIND SPOT, two weights in yolov8n. The `x == 0` clause readmits
        exact zeros, because a unit holding a legitimate zero weight would
        otherwise fail its own floor. A unit holding BOTH 0.0 and 2.0 then
        cannot tell a real zero from a 2.0 whose exponent MSB was cleared:
        model.22.dfl.conv holds the integers 0..15, so at bit 30 both
        0.0 -> 2.0 and 2.0 -> 0.0 go undetected. That is inherent to a
        magnitude-only bound rather than a defect here -- no [emin, emax] test
        can separate those two values -- and it is why the coverage census
        prints 100.00% at bit 30 instead of 99.999937%.
        """
        e = None
        if self.is_signed[name]:
            r = (x >= self.lo[name]) & (x <= self.hi[name])
        else:
            e = (x.view(torch.int32) >> 23) & 0xFF
            r = e <= self.emax[name]
        if self.is_two_sided[name]:
            if e is None:
                e = (x.view(torch.int32) >> 23) & 0xFF
            r = r & ((e >= self.emin[name]) | (x == 0))
        return r


def correct(w, bounds, name):
    """Algorithm 1, vectorised. Returns (repaired, flagged, toggled, clamped)."""
    ok = bounds.in_range(name, w)
    bad = ~ok
    n_bad = int(bad.sum())
    if n_bad == 0:
        return w, 0, 0, 0

    u = w.view(torch.int32)
    out = w.clone()
    done = torch.zeros_like(bad)

    for bit in EXP_BITS:                      # MSB to LSB, first hit wins
        cand = (u ^ (1 << bit)).view(torch.float32)
        hit = bad & ~done & bounds.in_range(name, cand)
        out = torch.where(hit, cand, out)
        done |= hit

    neg = -w                                  # Algorithm 1's sign retry
    hit = bad & ~done & bounds.in_range(name, neg)
    out = torch.where(hit, neg, out)
    done |= hit
    n_tog = int(done.sum())

    rem = bad & ~done                         # nothing worked: clamp
    if bool(rem.any()):
        if bounds.is_signed[name]:
            cl = torch.clamp(w, bounds.lo[name], bounds.hi[name])
        else:
            # magnitude-only: rebuild at the unit's ceiling, keeping the sign
            mag = (bounds.emax[name].to(torch.int32) << 23) | 0x7FFFFF
            cl = torch.sign(w) * mag.view(torch.float32)
            cl = torch.where(torch.isfinite(cl), cl, torch.zeros_like(cl))
        out = torch.where(rem, cl, out)
    return out, n_bad, n_tog, int(rem.sum())


class Stats:
    __slots__ = ("flagged", "toggled", "clamped", "exact", "false", "checks",
                 "b_flagged", "b_clamped", "b_exact", "b_false")

    def __init__(self):
        self.flagged = self.toggled = self.clamped = 0
        self.exact = self.false = self.checks = 0
        self.b_flagged = self.b_clamped = self.b_exact = self.b_false = 0

    def __repr__(self):
        return (f"checks={self.checks} w[flagged={self.flagged} "
                f"exact={self.exact} false={self.false} clamped={self.clamped}] "
                f"b[flagged={self.b_flagged} exact={self.b_exact} "
                f"false={self.b_false} clamped={self.b_clamped}]")


def attach(core, bounds, commit=True, golden=None):
    """Register the checker ahead of every conv.

    A forward_pre_hook is the faithful place for it: the paper checks weights
    on the fetch path, before the MACs consume them, and writes repairs back
    ("the corrected weight is used and committed to memory"), which is what
    commit=True does. Pass golden={name: tensor} to classify each repair as
    exact or false. Returns (handles, stats); call detach(handles) after.
    """
    st = Stats()
    handles = []

    def make(name):
        bkey = name + "#bias"

        def hook(mod, args):
            st.checks += 1
            tgts = [(name, mod.weight.data, False)]
            if mod.bias is not None and bkey in bounds.units:
                tgts.append((bkey, mod.bias.data, True))
            for key, t, is_bias in tgts:
                fixed, nb, ntog, ncl = correct(t, bounds, key)
                if not nb:
                    continue
                if is_bias:
                    st.b_flagged += nb
                    st.b_clamped += ncl
                else:
                    st.flagged += nb
                    st.toggled += ntog
                    st.clamped += ncl
                if golden is not None and key in golden:
                    g = golden[key]
                    changed = fixed.view(torch.int32) != t.view(torch.int32)
                    same = fixed.view(torch.int32) == g.view(torch.int32)
                    ex = int((changed & same).sum())
                    fa = int((changed & ~same).sum())
                    if is_bias:
                        st.b_exact += ex
                        st.b_false += fa
                    else:
                        st.exact += ex
                        st.false += fa
                if commit:
                    t.copy_(fixed)
            return None
        return hook

    for name, m in core.named_modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_pre_hook(make(name)))
    return handles, st


def detach(handles):
    for h in handles:
        h.remove()


def snapshot(core):
    """Pristine copy of weights AND biases, for exact-vs-false accounting.
    Keyed to match Bounds: "<layer>" for weights, "<layer>#bias" for biases."""
    out = {}
    for n, m in core.named_modules():
        if isinstance(m, nn.Conv2d):
            out[n] = m.weight.data.clone()
            if m.bias is not None:
                out[n + "#bias"] = m.bias.data.clone()
    return out
