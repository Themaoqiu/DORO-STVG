#!/bin/bash
# Activate virtual environment for STVG
# Usage: source scripts/activate_env.sh /path/to/your/env

if [ -z "$1" ]; then
    echo "❌ Error: Please specify environment path"
    echo "Usage: source $0 /path/to/your/env"
    echo ""
    echo "Example:"
    echo "  source $0 ~/.virtualenvs/stvg/dev"
    return 1 2>/dev/null || exit 1
fi

ENV_PATH="$1"

if [ ! -d "${ENV_PATH}" ]; then
    echo "❌ Error: Environment not found at ${ENV_PATH}"
    echo ""
    echo "Create it first with:"
    echo "  ./scripts/setup_env.sh ${ENV_PATH}"
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "${ENV_PATH}/bin/activate" ]; then
    echo "❌ Error: Not a valid virtual environment"
    return 1 2>/dev/null || exit 1
fi

echo "🚀 Activating environment: ${ENV_PATH}"
source "${ENV_PATH}/bin/activate"
