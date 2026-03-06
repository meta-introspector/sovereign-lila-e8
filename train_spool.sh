#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "🔮 Training LILA-E8 on Spool Dataset"
echo "   Using cached filelist: spool_filelist.json"
echo ""

# Build filelist cache if missing
if [ ! -f spool_filelist.json ]; then
    echo "📂 Building spool filelist cache..."
    python scripts/build_spool_filelist.py
fi

nix develop --impure --command python scripts/train_model.py \
  --checkpoint_dir checkpoints_spool \
  --resume
