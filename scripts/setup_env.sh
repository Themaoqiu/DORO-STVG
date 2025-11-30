#!/bin/bash
# Create virtual environment for STVG
# Usage: ./scripts/setup_env.sh /path/to/your/env

if [ -z "$1" ]; then
    echo "❌ Error: Please specify environment path"
    echo "Usage: $0 /path/to/your/env"
    echo ""
    echo "Example:"
    echo "  $0 ~/.virtualenvs/stvg/dev"
    exit 1
fi

ENV_PATH="$1"

echo "🔧 Creating virtual environment..."
echo "📂 Path: ${ENV_PATH}"

# Create environment with Python 3.11
uv venv "${ENV_PATH}" --python 3.11

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Environment created successfully!"
    echo ""
    echo "📝 Next steps:"
    echo "  1. Activate: source scripts/activate_env.sh ${ENV_PATH}"
    echo "  2. Install:  uv sync --active"
    echo ""
else
    echo "❌ Failed to create environment"
    exit 1
fi
