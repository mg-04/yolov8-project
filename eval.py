from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")

# Run evaluation on a small built-in dataset (auto-downloads a tiny 8-image COCO subset)
metrics = model.val(data="coco8.yaml", device=0)

# Print key results
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")