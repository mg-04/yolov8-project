#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Emit RTL test vectors from src/weightSentry.py itself.

The RTL has to agree with the exact Python that produced every number in this
project, so vectors come from real yolov8n weights and their real per-unit
bounds rather than from a reimplementation of the rule. Each line is

    w_in emax emin lo hi expected_flag

in hex; fields a mode does not use are emitted as zero.

in_range() is evaluated on the WHOLE tensor and the result is then indexed.
Its bound tables are expanded to tensor shape, so handing it a single element
would broadcast and judge every site against element 0's bound.

FP32 only: Bounds hardcodes the FP32 exponent field, so FP16 equivalence waits
on porting the Python side.

Usage:  ./gen_vectors.py [outdir]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))        # hw/verif/src
HW   = os.path.dirname(os.path.dirname(HERE))            # hw
ROOT = os.path.dirname(HW)                               # project root
sys.path.insert(0, os.path.join(ROOT, "src"))
from ultralytics import YOLO                                    # noqa: E402
import weightSentry as WS                                       # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HW, "verif", "data")
MODES = [("exp1", 0, dict(signed=False, two_sided=False)),
         ("exp2", 1, dict(signed=False, two_sided=True)),
         ("sgn",  2, dict(signed=True,  two_sided=False)),
         ("sgnf", 3, dict(signed=True,  two_sided=True))]
BITS = list(range(31, 22, -1))
KEY_STRIDE = 11          # spread over layers rather than dumping one tensor
SITES = 16               # sites sampled per tensor


def as_u32(t):
    return (t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).tolist()


def tensor_for(core_modules, key):
    mod = core_modules[key.replace("#bias", "")]
    return (mod.bias if key.endswith("#bias") else mod.weight).data


def main():
    model = YOLO(os.path.join(ROOT, "yolov8n.pt"))
    model.model.fuse()
    core = model.model.eval()
    mods = dict(core.named_modules())
    os.makedirs(OUT, exist_ok=True)

    for tag, mode, cfg in MODES:
        b = WS.Bounds(core, mode="chan", **cfg)
        rows, n_flag = [], 0
        for key in list(b.units)[::KEY_STRIDE]:
            t = tensor_for(mods, key)
            n = t.numel()
            idx = torch.arange(0, n, max(1, n // SITES))[:SITES]

            def pick(table, default=0):
                if not table:
                    return [default] * len(idx)
                return table[key].reshape(-1)[idx].tolist()

            emax, emin = pick(b.emax), pick(b.emin)
            lo = as_u32(b.lo[key].reshape(-1)[idx]) if b.lo else [0] * len(idx)
            hi = as_u32(b.hi[key].reshape(-1)[idx]) if b.hi else [0] * len(idx)

            for bit in [None] + BITS:
                w = t if bit is None else \
                    (t.view(torch.int32) ^ (1 << bit)).view(torch.float32)
                ok = b.in_range(key, w).reshape(-1)       # full tensor, then index
                wi = as_u32(w.reshape(-1)[idx])
                for j, site in enumerate(idx.tolist()):
                    f = 0 if bool(ok[site]) else 1
                    n_flag += f
                    rows.append(
                        f"{wi[j]:08x} {int(emax[j]):02x} {int(emin[j]):02x} "
                        f"{int(lo[j]):08x} {int(hi[j]):08x} {f}")

        path = os.path.join(OUT, f"{tag}.vec")
        with open(path, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        print(f"{tag:5s} MODE={mode}  {len(rows):>7,} vectors, "
              f"{n_flag:>7,} expect a flag  ->  {path}")


if __name__ == "__main__":
    main()
