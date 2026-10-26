#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Ladder step 1: prove the partition is numerically invisible.

Runs the same images three ways and demands BIT-IDENTICAL detections:

  monolithic   the model as shipped
  rung 1a      model.0 and model.22.dfl replaced by stubs calling in-process
  rung 1b      same stubs, bytes over a loopback socket

Bit-identical is the right bar, not "close". Later rungs move this computation
to ARM cores and then to FPGA fabric, where real numerical differences WILL
appear -- and the only way to attribute them to the kernel is to know the
partition itself contributes nothing. Approximate agreement here would leave
every later discrepancy unattributable.

Usage:
  ./analysis/partition/split_test.py [images]
"""
import logging, os, sys, threading, time

import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import fi_lib                                                    # noqa: E402
import shield                                                    # noqa: E402

logging.getLogger("ultralytics").setLevel(logging.ERROR)

IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 64
DATA = "coco-val2017-sub1024.yaml"
PORT = 9801
# model.0 holds 43.3% of the |w| >= 2 population in 448 values; dfl is 16
# values that are 87.5% such. Both sit at a network boundary, so the trust
# crossing coincides with I/O that happens anyway.
TAGS = ["model.0", "model.22.dfl"]


def load():
    m = YOLO(os.path.join(ROOT, "yolov8n.pt"))
    m.model.fuse()
    return m.model.cuda().eval()


def detect_all(core, paths, warm=True):
    """Time only steady state. make_detector caches preprocessed images, and the
    first CUDA call pays context init plus cuDNN autotune, so an untimed pass
    over the full set is required before the numbers mean anything."""
    det = fi_lib.make_detector(core)
    if warm:
        for p in paths:
            det(p)
        torch.cuda.synchronize()
    t0 = time.time()
    out = {p: det(p)[0] for p in paths}
    torch.cuda.synchronize()
    return out, time.time() - t0


def compare(gold, test, label):
    """Exact comparison. Counts rows that match to the last bit."""
    nd = nb = exact = 0
    dbox = dconf = 0.0
    for p, g in gold.items():
        t = test[p]
        if len(g) != len(t):
            nd += 1
            continue
        for a, b in zip(g, t):
            nb += 1
            if a[0] != b[0]:
                continue
            dbox = max(dbox, max(abs(x - y) for x, y in zip(a[2], b[2])))
            dconf = max(dconf, abs(a[1] - b[1]))
            if a[1] == b[1] and all(x == y for x, y in zip(a[2], b[2])):
                exact += 1
    ok = nd == 0 and dbox == 0.0 and dconf == 0.0
    print(f"  {label:28s} count-mismatch {nd:3d}   max|box| {dbox:.3e}   "
          f"max|conf| {dconf:.3e}   bit-exact {exact}/{nb}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


paths = fi_lib.dataset_images(DATA, IMAGES)
print(f"{len(paths)} images, splitting {TAGS}\n")

core = load()
GOLD, mono_s = detect_all(core, paths)
print(f"monolithic: {sum(len(v) for v in GOLD.values())} detections, "
      f"{1000*mono_s/len(paths):.2f} ms/image\n")

# ---- rung 1a: in-process ---------------------------------------------------
core = load()
real = shield.split(core, TAGS)
shield.attach(core, shield.LocalTransport(real))
out, a_s = detect_all(core, paths)
a_ok = compare(GOLD, out, "1a in-process")

# ---- rung 1b: loopback socket ---------------------------------------------
core = load()
real = shield.split(core, TAGS)

# start remote server, using the splitted out weights `real` as dict args
srv = threading.Thread(target=shield.serve,
                       kwargs=dict(modules=real, host="127.0.0.1", port=PORT),
                       daemon=True)
srv.start()
time.sleep(1.0)
tr = shield.SocketTransport("127.0.0.1", PORT)
shield.attach(core, tr)

# run detection on the modified `core`
out, b_s = detect_all(core, paths)
b_ok = compare(GOLD, out, "1b loopback socket")
tr.close()

MB = (3*640*640 + 16*320*320 + 64*8400 + 4*8400) * 4 / 1e6
mono, a, b = (1000*x/len(paths) for x in (mono_s, a_s, b_s))
print(f"\nsteady-state latency, ms/image")
print(f"  monolithic          {mono:6.2f}")
print(f"  1a in-process       {a:6.2f}   ({a-mono:+.2f} = cost of the cut itself)")
print(f"  1b loopback socket  {b:6.2f}   ({b-a:+.2f} = cost of serialisation)")
print(f"\nboundary traffic {MB:.2f} MB/frame"
      f"  ->  loopback moved it at {MB/(b-a)*1000/1000:.2f} GB/s"
      if b > a else "")
sys.exit(0 if (a_ok and b_ok) else 1)
