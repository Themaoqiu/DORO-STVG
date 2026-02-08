# Shot Scene Graph generator

## Virtual environment

### yolo and sam3
```bash
cd graph_generator
uv venv .venv/main --python 3.11
source .venv/main/bin/activate
uv sync
```

### action detector
```bash
uv venv .venv/mmaction --python 3.11
source .venv/mmaction/bin/activate
uv pip install -r requirements_action.txt
```

## Run generator

```bash
bash scripts/run_generator.sh
```

## Run visualizer
```bash
bash scripts/run_visualizer.sh
```