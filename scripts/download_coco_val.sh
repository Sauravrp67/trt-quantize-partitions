#!/usr/bin/env bash
set -euo pipefail
DEST="$(dirname "$0")/../data/coco"
mkdir -p "$DEST"
cd "$DEST"
[ -d val2017 ] || { curl -L -O http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip && rm val2017.zip; }
[ -f annotations/instances_val2017.json ] || { curl -L -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip; }
echo "COCO val2017: $(ls val2017 | wc -l) images"