#!/usr/bin/env bash
# VisDrone2019-VID test-dev: 17 drone sequences with detection ground truth.
#
# Upstream ships JPEG frame sequences, not video files, so this also encodes
# one .mp4 per sequence for the infer CLI. FPS is not recorded anywhere in the
# release; 25 is an assumption, override with VISDRONE_FPS.
#
# The official Google Drive links are chronically quota-blocked ("too many
# users have viewed or downloaded this file recently"), so this pulls from a
# HuggingFace mirror of the same zip (2293.5 MB, matches upstream's 2.14 GiB).
set -euo pipefail
DEST="$(dirname "$0")/../data/test_videos/visdrone_vid"
FPS="${VISDRONE_FPS:-25}"
ZIP=VisDrone2019-VID-test-dev.zip
SEQ_ROOT="VisDrone2019-VID-test-dev/sequences"

mkdir -p "$DEST"
cd "$DEST"

[ -d "$SEQ_ROOT" ] || {
  [ -f "$ZIP" ] || hf download AndriiDemk/visDrone_copy "$ZIP" --repo-type dataset --local-dir .
  unzip -q "$ZIP" && rm -f "$ZIP"
  rm -rf .cache
}

mkdir -p videos
for seq in "$SEQ_ROOT"/*/; do
  name="$(basename "$seq")"
  [ -f "videos/$name.mp4" ] && continue
  ffmpeg -loglevel error -y -framerate "$FPS" -start_number 1 -i "$seq/%07d.jpg" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 18 -pix_fmt yuv420p \
    "videos/$name.mp4"
  echo "  encoded $name"
done

echo "VisDrone-VID test-dev: $(ls "$SEQ_ROOT" | wc -l) sequences, $(ls videos/*.mp4 | wc -l) videos @ ${FPS}fps"
