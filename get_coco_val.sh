#!/bin/bash
# Download COCO val2017 (5,000 held-out images) WITHOUT the 19 GB train2017 set.
# Replicates the layout Ultralytics' coco.yaml expects.
set -e

DEST=/home/mgong2/datasets
mkdir -p "$DEST"
cd "$DEST"

echo "[1/4] labels (46 MB) -> $DEST/coco/labels"
curl -L --fail -o labels.zip \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
unzip -q -o labels.zip
rm labels.zip

echo "[2/4] val2017 images (778 MB) -> $DEST/coco/images/val2017"
mkdir -p coco/images
curl -L --fail -o coco/images/val2017.zip \
  http://images.cocodataset.org/zips/val2017.zip

echo "[3/4] extracting images"
cd coco/images
unzip -q -o val2017.zip
rm val2017.zip

echo "[4/4] verifying"
cd "$DEST/coco"
imgs=$(find images/val2017 -name '*.jpg' | wc -l)
lbls=$(find labels/val2017 -name '*.txt' | wc -l)
echo "  images: $imgs (expect 5000)"
echo "  labels: $lbls"
echo "  val list: $(wc -l < val2017.txt) entries"
du -sh "$DEST/coco"
echo "DONE"
