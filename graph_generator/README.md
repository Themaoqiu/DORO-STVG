# Shot Scene Graph generator

## Install dependencies

### yolo and sam3

Install yolo and sam3 dependencies.
```bash
cd graph_generator
uv venv .venv/main --python 3.11
source .venv/main/bin/activate
uv sync
```

Download yolo weights and sam3 checkpoint.
```bash
hf download Ultralytics/YOLO11 yolo11x.pt --local-dir /path
hf download facebook/sam3 --local-dir /path
```

### action detector

Install action detector dependencies.
```bash
uv venv .venv/mmaction --python 3.11
source .venv/mmaction/bin/activate
uv pip install -r requirements_action.txt
```

Download action detector checkpoint.
```bash
wget https://download.openmmlab.com/mmaction/v1.0/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-bf93c9ea.pth -O vit-large-p16_videomae-k400-pre.pth
```

## Run generator

```bash
bash scripts/run_generator.sh
```

## Run visualizer
```bash
bash scripts/run_visualizer.sh
```