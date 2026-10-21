import logging, time, sys, os, torch
logging.getLogger("ultralytics").setLevel(logging.ERROR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import fi_lib
from ultralytics import YOLO

p = torch.cuda.get_device_properties(0)
print("  device: %s  cc %d.%d" % (p.name, p.major, p.minor))
print("  cudnn.allow_tf32 : %s" % torch.backends.cudnn.allow_tf32)
print("  matmul.allow_tf32: %s" % torch.backends.cuda.matmul.allow_tf32)

m = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")); m.model.fuse()
core = m.model.cuda().eval()
detect = fi_lib.make_detector(core)
imgs = fi_lib.dataset_images("coco-val2017-sub1024.yaml", 32)

GOLD = {q: detect(q)[0] for q in imgs}                 # warms the image cache
t0 = time.time(); GOLD = {q: detect(q)[0] for q in imgs}
g = (time.time()-t0)/len(imgs)*1000
ndet = sum(len(v) for v in GOLD.values())

layer = dict(core.named_modules())["model.7.conv"]
flat = layer.weight.data.view(-1); orig = flat[12345].clone()
flat[12345] = fi_lib.flip(flat[12345], 30)
agg = {"masked":0,"benign":0,"SDC":0,"DUE":0}
t0 = time.time()
for q in imgs:
    d, nn, ni = detect(q); agg[fi_lib.outcome(GOLD[q], d, nn, ni)[0]] += 1
c = (time.time()-t0)/len(imgs)*1000
flat[12345] = orig
print("  golden   : %7.2f ms/image  (%d detections)" % (g, ndet))
print("  corrupted: %7.2f ms/image" % c)
print("  verdicts : %s" % agg)
