#!/bin/bash
# One-time setup: this folder's own Python environment, and the SAM 2.1 weights (about 900 MB, optional).
#   ./setup.sh            everything
#   ./setup.sh --no-sam   without PyTorch, SAM and the landmark models (select_object then uses a colour graph cut)
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "needs uv: brew install uv" >&2; exit 1; }
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
if [ "${1:-}" != "--no-sam" ]; then
  uv pip install --python .venv/bin/python "torch>=2.5.1" "torchvision>=0.20.1" "sam2==1.1.0"
  mkdir -p .models
  if [ ! -f .models/sam2.1_hiera_large.pt ]; then
    curl -L --fail -o .models/sam2.1_hiera_large.pt.part https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    mv .models/sam2.1_hiera_large.pt.part .models/sam2.1_hiera_large.pt
  fi
fi
# Landmark models (pose, face, hands) for measuring figures: MediaPipe, about 42 MB. Optional like SAM.
if [ "${1:-}" != "--no-sam" ]; then
  uv pip install --python .venv/bin/python mediapipe
  # MediaPipe pulls in a second OpenCV build; keep only the headless one with the contrib modules.
  uv pip uninstall --python .venv/bin/python opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
  uv pip install --python .venv/bin/python "opencv-contrib-python-headless>=4.10"
  mkdir -p .models
  for u in pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task face_landmarker/face_landmarker/float16/latest/face_landmarker.task hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task; do
    f=".models/$(basename "$u")"
    [ -f "$f" ] || curl -L --fail -o "$f" "https://storage.googleapis.com/mediapipe-models/$u"
  done
fi
echo "ready: ./vw photo.jpg \"what to change\"   ·   ./vw \"what to draw\"   ·   ./vw --help"
