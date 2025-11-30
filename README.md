# DORO-STVG

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. Create virtual environment

Choose your environment path and create it:

```bash
bash ./scripts/setup_env.sh ~/.virtualenvs/stvg/dev
```

### 3. Activate environment

```bash
source scripts/activate_env.sh ~/.virtualenvs/stvg/dev
```

### 4. Install package

```bash
uv sync --active
```

### 5. Configuration for video processing

Set the video reader backend by exporting the environment variable:

```bash
# Use torchvision (default)
export FORCE_QWENVL_VIDEO_READER=torchvision

# Or use decord (recommend)
export FORCE_QWENVL_VIDEO_READER=decord

# Or use torchcodec
export FORCE_QWENVL_VIDEO_READER=torchcodec
```

### 6. Add and remove dependencies

```bash
# Add a new package
uv add --active [package-name]

# Remove a package
uv remove --active [package-name]
```

### 7. Deactivate environment

```bash
deactivate
```

### 8. Switch to different environment

```bash
# Deactivate current environment
deactivate

# Activate another environment
source scripts/activate_env.sh /path/to/another/env
```

## Quick Start

```bash
python
```