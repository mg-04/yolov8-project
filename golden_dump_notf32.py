import logging, sys, os, json, torch
logging.getLogger("ultralytics").setLevel(logging.ERROR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import torch
torch.backends.cudnn.allow_tf32 = False
import fi_lib
from ultralytics import YOLO
m = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")); m.model.fuse()
detect = fi_lib.make_detector(m.model.cuda().eval())
out = {}
for p in fi_lib.dataset_images("coco-val2017-sub1024.yaml", 32):
    out[os.path.basename(p)] = [[d[0], round(d[1], 6), [round(v, 4) for v in d[2]]]
                                for d in detect(p)[0]]
json.dump(out, open(sys.argv[1], "w"), sort_keys=True)
print("  wrote", sys.argv[1], "-", sum(len(v) for v in out.values()), "detections")
