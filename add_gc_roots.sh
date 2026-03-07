#!/usr/bin/env bash
# Add GC roots for the nix environment to prevent garbage collection

set -e
cd "$(dirname "$0")"

echo "🔒 Adding GC roots for LILA-E8 environment..."

# Create gc-roots directory if it doesn't exist
mkdir -p .gc-roots

# Build the environment and create a GC root
nix develop --impure --profile .gc-roots/lila-e8-env

echo "✅ GC root created at .gc-roots/lila-e8-env"
echo "   This environment will not be garbage collected"
