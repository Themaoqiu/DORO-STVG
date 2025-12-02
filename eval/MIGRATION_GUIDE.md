# 从旧版迁移到新版

## 快速对照表

### 旧版 → 新版文件映射

| 旧版文件 | 新版文件 | 说明 |
|---------|---------|------|
| `eval/main.py` | `main.py` | Fire CLI 入口 |
| `eval/runner.py` | `pipelines/hcstvg.py` | 评测流程 |
| `eval/model_registry/qwen_family.py` | `models/qwen.py` | 模型封装 |
| `eval/evaluator/metrics.py` | `utils/metrics.py` | 指标计算 |
| `eval/prompts/stvg.py` | `prompts.py` | Prompt 定义 |
| `eval/config/hcstvg_qwen.yaml` | **删除** | 改用 bash 参数 |
| `eval/core/schema.py` | **删除** | 简化数据结构 |
| `eval/dataset_registry/` | **删除** | 集成到 pipeline |

### 命令对照

#### 旧版运行方式
```bash
# 需要编辑 YAML 配置文件
vim eval/config/hcstvg_qwen.yaml

# 运行评测
python eval/main.py
```

#### 新版运行方式
```bash
# 方式 1: Bash 脚本 (推荐)
./run_eval.sh \
  --model_name qwen2.5vl \
  --model_path /path/to/model \
  --annotation_path /data/hcstvg/test.json \
  --video_dir /data/hcstvg/videos

# 方式 2: Fire CLI
python main.py run_hcstvg \
  --model_name=qwen2.5vl \
  --model_path=/path/to/model \
  --annotation_path=/data/hcstvg/test.json \
  --video_dir=/data/hcstvg/videos
```

## 代码查找指南

### 想找某个功能？直接看这里！

| 功能 | 旧版位置 | 新版位置 |
|------|---------|---------|
| **数据加载** | `eval/dataset_registry/hcstvg_dataset.py` | `pipelines/hcstvg.py` → `load_data()` |
| **帧号映射** | `eval/dataset_registry/hcstvg_dataset.py` | `pipelines/hcstvg.py` → `_get_frame_mapping()` |
| **视频抽帧** | `utils/stvg_video_utils.py` | `utils/stvg_video_utils.py` (不变) |
| **模型推理** | `eval/model_registry/qwen_family.py` | `models/qwen.py` → `predict_batch()` |
| **Prompt 格式化** | `eval/prompts/stvg.py` | `prompts.py` → `format_stvg_prompt()` |
| **响应解析** | `eval/prompts/stvg.py` | `prompts.py` → `parse_stvg_response()` |
| **指标计算** | `eval/evaluator/metrics.py` | `utils/metrics.py` (不变) |
| **结果保存** | `eval/runner.py` | `pipelines/hcstvg.py` → `_save_results()` |
| **完整流程** | `eval/runner.py` | `pipelines/hcstvg.py` → `run_evaluation()` |

### 追踪变量计算逻辑

#### 例子 1: 想知道 `gt_temporal_sampled` 怎么计算的？

**新版 (一步到位)**:
1. 打开 `pipelines/hcstvg.py`
2. 搜索 `gt_temporal_sampled`
3. 在 `load_data()` 方法中找到:
```python
gt_temporal_sampled = (
    self._map_frame_to_sampled(st_frame_orig, sampled_indices),
    self._map_frame_to_sampled(ed_frame_orig, sampled_indices)
)
```
4. 查看 `_map_frame_to_sampled()` 方法了解映射逻辑

**旧版 (需要跳转多次)**:
1. `eval/runner.py` → 调用 dataset
2. `eval/dataset_registry/registry.py` → 查找 dataset 类
3. `eval/dataset_registry/hcstvg_dataset.py` → 查看数据加载
4. `eval/core/schema.py` → 理解数据结构
5. 最终才找到计算逻辑

#### 例子 2: 想知道模型怎么推理的？

**新版**:
1. 打开 `pipelines/hcstvg.py` → `_process_batch()`
2. 看到调用 `self.model.predict_batch()`
3. 打开 `models/qwen.py` → `predict_batch()` 方法
4. 完整的 vllm 推理逻辑就在这里

**旧版**:
1. `eval/runner.py` → 调用 model
2. `eval/model_registry/registry.py` → 查找 model 类
3. `eval/model_registry/base_model.py` → 理解基类
4. `eval/model_registry/qwen_family.py` → 最终找到推理逻辑

## 配置迁移示例

### 旧版 YAML 配置
```yaml
# eval/config/hcstvg_qwen.yaml
model:
  name: "qwen2.5vl"
  model_path: "/path/to/Qwen2.5-VL-7B"
  batch_size: 1
  nframes: 100
  max_tokens: 512
  tensor_parallel_size: 1

dataset:
  name: "hcstvg"
  annotation_path: "/data/hcstvg/test.json"
  video_dir: "/data/hcstvg/videos"

output:
  save_dir: "./results"
```

### 新版 Bash 参数
```bash
./run_eval.sh \
  --model_name qwen2.5vl \
  --model_path /path/to/Qwen2.5-VL-7B \
  --batch_size 1 \
  --num_frames 100 \
  --max_tokens 512 \
  --tensor_parallel_size 1 \
  --annotation_path /data/hcstvg/test.json \
  --video_dir /data/hcstvg/videos \
  --output_dir ./results
```

## 关键改进点

### 1. 不再需要 Registry 模式
- ❌ 旧版: 需要注册 model 和 dataset
- ✅ 新版: 直接在 `main.py` 中选择模型类

### 2. 数据结构简化
- ❌ 旧版: `STVGSample` 和 `Result` 类
- ✅ 新版: 直接使用字典，更灵活

### 3. 配置方式改进
- ❌ 旧版: 静态 YAML 文件
- ✅ 新版: 动态命令行参数

### 4. 帧号映射集中化
- ❌ 旧版: 分散在 dataset、schema、runner 中
- ✅ 新版: 集中在 `pipelines/hcstvg.py` 中

## 常见问题 FAQ

### Q1: 如何添加新的模型 (如 LLaVA)?
**新版**:
1. 在 `models/` 创建 `llava.py`
2. 实现 `predict_batch()` 方法
3. 在 `main.py` 的 `run_hcstvg()` 中添加模型选择逻辑

### Q2: 如何添加新的数据集 (如 VidSTG)?
**新版**:
1. 在 `pipelines/` 创建 `vidstg.py`
2. 实现完整评测流程 (参考 `hcstvg.py`)
3. 在 `main.py` 添加 `run_vidstg()` 方法

### Q3: 如何修改 Prompt?
**新版**:
直接编辑 `prompts.py` 中的字符串常量

### Q4: 如何查看某个变量的计算过程?
**新版**:
1. 在 VSCode 中打开 `pipelines/hcstvg.py`
2. 使用 Cmd+F (Mac) 或 Ctrl+F (Win) 搜索变量名
3. 单文件内就能看到完整逻辑

### Q5: 如何清理磁盘空间?
**新版**:
```bash
# 方式 1
./run_eval.sh --cleanup_only

# 方式 2
python main.py cleanup

# 方式 3: 评测后自动清理
./run_eval.sh ... --cleanup_after
```

## 迁移检查清单

- [ ] 安装 `fire` 包: `pip install fire`
- [ ] 确认 `run_eval.sh` 有执行权限: `chmod +x run_eval.sh`
- [ ] 测试运行: `./run_eval.sh --help`
- [ ] 删除旧的 YAML 配置文件 (已不需要)
- [ ] 更新自己的评测脚本为新的命令行方式
- [ ] 阅读 `REFACTOR_README.md` 了解完整功能

## 总结

**核心变化**:
- 🗂️ 目录结构: 4层嵌套 → 2层扁平
- ⚙️ 配置方式: YAML → Bash 参数
- 🔍 代码追踪: 多次跳转 → 单文件查看
- 📝 命名规范: 统一 `gt_`, `pred_`, `fullresponse`
- 🎯 帧号映射: 分散 → 集中
- 💾 空间管理: 累积 → 覆盖 + 清理

**优势**:
- ✅ 代码可读性大幅提升
- ✅ 调试和追踪更容易
- ✅ 配置更灵活
- ✅ 扩展更简单
- ✅ 维护成本更低

欢迎使用新版架构！🎉
