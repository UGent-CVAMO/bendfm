#!/usr/bin/env bash
# BenDFM setup script: initializes openpoints submodule for PointNeXt (optional)
# This script is only needed if using PointNeXt. UV-Net requires no additional setup.

set -e

echo "=== BenDFM Setup (OpenPoints for PointNeXt) ==="
echo ""

# Update git submodules
echo "Initializing git submodules..."
git submodule update --init --recursive
echo "✓ Submodules initialized"
echo ""

echo "Setup complete! You can now train models:"
echo ""
echo "UV-Net (recommended - no additional setup):"
echo "  python benchmark/train/uvnet.py --data_dir data/bendfm --label_key bbox_area_unfolded --regression"
echo ""
echo "PointNeXt (requires openpoints configuration):"
echo "  Follow OpenPoints documentation: https://github.com/guochengqian/openpoints#installation"
echo "  Then: python benchmark/train/pointnext.py --data_dir data/bendfm --label_key bbox_area_unfolded --regression"
echo ""
echo "=== Setup Complete ==="
echo "You can now train models:"
echo "  python benchmark/train/pointnext.py --data_dir data/bendfm --label_key bbox_area_unfolded --regression"
echo "  python benchmark/train/uvnet.py --data_dir data/bendfm --label_key bbox_area_unfolded --regression"
