#!/usr/bin/env bash
set -e

echo "Installing system packages..."
apt update
apt install -y \
  curl \
  git \
  ffmpeg \
  nodejs \
  npm \
  gh \
  bubblewrap \
  cudnn9-cuda-12

echo "Installing Python packages..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Needed for YouTube livestream resolving
python -m pip install --upgrade yt-dlp

# Needed for CUDA/GPU inference
python -m pip install --upgrade onnxruntime-gpu

echo "Installing OpenAI Codex CLI..."
npm install -g @openai/codex@latest

echo "Refreshing dynamic linker cache..."
ldconfig

echo
echo "ONNX Runtime providers:"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

echo
echo "yt-dlp version:"
yt-dlp --version

echo
echo "Node version:"
node -v

echo
echo "npm version:"
npm -v

echo
echo "Codex version:"
codex --version || true

echo
echo "GitHub CLI version:"
gh --version || true

echo
echo "Dev setup complete."
echo
echo "Next manual steps:"
echo "1. Authenticate GitHub:"
echo "   gh auth login"
echo
echo "2. Authenticate Codex:"
echo "   codex"
echo
echo "3. Configure git identity if needed:"
echo "   git config --global user.name \"LowkeyMad\""
echo "   git config --global user.email \"wafflesandteashirt@gmail.com\""
