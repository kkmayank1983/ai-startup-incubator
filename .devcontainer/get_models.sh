#!/usr/bin/env bash
# Downloads the teacher-trained models that are too big to keep in git.
# They live on the repo's "models" release page:
#   https://github.com/kkmayank1983/ai-startup-incubator/releases/tag/models
#
# Runs once when a codespace is created (from setup.sh). Safe to run again
# at any time:   bash .devcontainer/get_models.sh
# It skips any model you already have.
set -e
cd "$(dirname "$0")/.."

BASE="https://github.com/kkmayank1983/ai-startup-incubator/releases/download/models"
MODELS="qa_model_v7.pth qa_model_v8.pth"

for f in $MODELS; do
  if [ -f "$f" ]; then
    echo "   $f is already here"
  else
    echo "   downloading $f ..."
    curl -fL --retry 3 --progress-bar -o "$f.part" "$BASE/$f"
    mv "$f.part" "$f"
  fi
done
echo "   models ready: $MODELS"
