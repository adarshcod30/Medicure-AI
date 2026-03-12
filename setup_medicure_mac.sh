#!/bin/bash
# ============================================================
#  Medicure-AI : macOS One-Time Setup Script
#  Author : Adarsh Dwivedi
#  Purpose: Configure Python environment, VS Code, Git identity
# ============================================================

echo "🚀 Starting Medicure-AI setup on macOS..."

# ---------- 1. Verify Homebrew ----------
if ! command -v brew &>/dev/null; then
  echo "🍺 Homebrew not found — installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  echo "✅ Homebrew found."
fi

# ---------- 2. Install dependencies ----------
echo "📦 Installing core tools (Git, Miniforge, VS Code)..."
brew install git
brew install --cask miniforge
brew install --cask visual-studio-code
brew install ghostscript  # required for Camelot

# ---------- 3. Create conda environment ----------
echo "🐍 Setting up Conda environment: medicure-ai"
source /opt/homebrew/Caskroom/miniforge/latest/bin/activate || true
if conda info --envs | grep -q medicure-ai; then
  echo "🔁 Environment 'medicure-ai' already exists. Skipping creation."
else
  conda create -n medicure-ai python=3.10 -y
fi

conda activate medicure-ai

# ---------- 4. Install Python dependencies ----------
echo "📚 Installing Python dependencies..."
# Create requirements.txt if missing
if [ ! -f "requirements.txt" ]; then
  cat > requirements.txt <<'EOF'
pandas
numpy
rapidfuzz
pdfplumber
camelot-py[cv]
openpyxl
matplotlib
EOF
fi

pip install -r requirements.txt

# ---------- 5. Configure Git identity ----------
echo "👤 Setting global Git identity..."
git config --global user.name "Adarsh Dwivedi"
git config --global user.email "your_github_email@example.com"
git config --global credential.helper osxkeychain

# ---------- 6. Create VS Code config ----------
mkdir -p .vscode
cat > .vscode/settings.json <<'EOF'
{
  "python.defaultInterpreterPath": "/opt/homebrew/Caskroom/miniforge/latest/envs/medicure-ai/bin/python",
  "python.analysis.autoImportCompletions": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "editor.formatOnSave": true
}
EOF

cat > .vscode/tasks.json <<'EOF'
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run merge_datasets",
      "type": "shell",
      "command": "python scripts/merge/merge_datasets.py",
      "group": "build",
      "presentation": { "reveal": "always" }
    }
  ]
}
EOF

# ---------- 7. Final summary ----------
echo "✅ Setup complete!"
echo "To start working:"
echo "  1️⃣  conda activate medicure-ai"
echo "  2️⃣  code /Users/adarsh/Desktop/Medicure-AI"
echo "  3️⃣  Run tasks or scripts via VS Code or terminal."
echo "-----------------------------------------------------"
