#!/usr/bin/env bash
# Runs once when a codespace is created. Builds the same environment
# the teacher's computer has, from requirements.txt.
set -e
cd "$(dirname "$0")/.."

echo "== Unpacking large data files =="
find . -name '*.xz' -not -path './.git/*' -not -path './venv/*' | while read -r packed; do
  out="${packed%.xz}"
  if [ ! -f "$out" ]; then
    echo "   $packed -> $out"
    python3 -c "import lzma, shutil, sys; shutil.copyfileobj(lzma.open(sys.argv[1]), open(sys.argv[2], 'wb'))" "$packed" "$out"
  fi
done

echo "== Building venv/ =="
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip --quiet
venv/bin/python -m pip install -r requirements.txt

echo
echo "== Setup complete. Try:  python torch_basics.py =="
