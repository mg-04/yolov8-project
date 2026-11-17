#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""What do the exponent bits of a trained weight actually look like?

This is the census that explains the per-bit damage profile. A bit flip does not
have an intrinsic severity -- it has a severity that depends on whether that bit
is currently SET in the value being hit:

    bit currently 0 -> flip SETS it   -> exponent rises -> magnitude EXPLODES
    bit currently 1 -> flip CLEARS it -> exponent falls -> magnitude COLLAPSES

An explosion injects a huge value into the MACs and wrecks the output. A collapse
drives the weight to ~0, which is just pruning it, and is benign. So the measured
per-bit SDC curve is predicted almost entirely by the fraction of weights in which
that bit is zero.

For the biased exponent e = 127 + floor(log2|x|), bit 30 carries 128 and bit 29
carries 64, so:

    |x| >= 2            -> e >= 128 -> bit30=1
    2^-63 <= |x| < 2    -> 64 <= e < 128 -> bit30=0, bit29=1   <-- nearly everything
    |x| < 2^-63         -> e < 64 -> bit30=0, bit29=0          <-- only true zeros
    |x| >= 2^65         -> bit30=1, bit29=1                    <-- never in a trained net

Usage:  ./exponent_census.py
"""
import logging, os, sys
import torch

logging.getLogger("ultralytics").setLevel(logging.ERROR)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src")
from ultralytics import YOLO

MODELS = sys.argv[1:] or ["yolov8n"]
for mname in MODELS:
    model = YOLO(f"{ROOT}/{mname}.pt")
    model.model.fuse()
    core = model.model.eval()
    pools = {"weight": [], "bias": []}
    for _, m in core.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            pools["weight"].append(m.weight.data.reshape(-1))
            if m.bias is not None:
                pools["bias"].append(m.bias.data.reshape(-1))
    pools = {k: torch.cat(v) for k, v in pools.items() if v}

    print(f"\n{'='*78}\n{mname}: "
          + ", ".join(f"{k}s {v.numel():,}" for k, v in pools.items()) + f"\n{'='*78}")

    # ---- joint distribution of (bit30, bit29) ----------------------------
    print("\njoint (bit30, bit29):")
    print(f"{'pattern':>9}{'meaning':>26}" + "".join(f"{k:>22}" for k in pools))
    rowfmt = {}
    for b30 in (0, 1):
        for b29 in (0, 1):
            cells = []
            for k, t in pools.items():
                u = t.view(torch.int32)
                s = (((u >> 30) & 1) == b30) & (((u >> 29) & 1) == b29)
                n = int(s.sum())
                cells.append(f"{n:>11,} {100.0*n/t.numel():>8.4f}%")
            meaning = {(0,0): "|x| < 2^-63 (zeros)", (0,1): "2^-63 <= |x| < 2",
                       (1,0): "2 <= |x| < 2^65", (1,1): "|x| >= 2^65"}[(b30,b29)]
            print(f"{f'{b30}{b29}':>9}{meaning:>26}" + "".join(f"{c:>22}" for c in cells))

    # ---- per-bit: how often is it 0, and what does a flip do? ------------
    print("\nper-bit, fraction CLEAR (a flip on a clear bit explodes the value):")
    print(f"{'bit':>4}{'exp step':>10}{'x factor':>12}"
          + "".join(f"{k+' clear':>16}" for k in pools))
    for bit in range(30, 22, -1):
        step = 1 << (bit - 23)
        cells = []
        for k, t in pools.items():
            u = t.view(torch.int32)
            clear = int((((u >> bit) & 1) == 0).sum())
            cells.append(f"{100.0*clear/t.numel():>15.4f}%")
        fac = f"2^{step}" if step >= 8 else f"x{2**step}"
        print(f"{bit:>4}{'+'+str(step):>10}{fac:>12}" + "".join(cells))
    del model, core, pools
print("\n(a 'clear' bit is one a flip can only SET -> explosion; "
      "a set bit can only be CLEARED -> collapse toward zero)")
