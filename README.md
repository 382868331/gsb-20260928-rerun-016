# 动画资源加工系统

这是一个从零搭建的 Python 动画处理管线。资源、骨架映射和处理批次分别存放在 `resources/`、`mappings/` 和 `batches/`，后续压缩、格式转换或其他处理器可以复用同一套批次与原子写入机制。

## 能力

- 支持不同骨长骨架：按双腿骨长比例缩放根部位移和脚部世界位置。
- 使用世界旋转增量而不是直接复制局部旋转，可处理不同的初始姿态。
- 通过双腿 IK 保持源动作脚部落点，避免只替换关节名称。
- 缺失映射、映射到不存在关节、必需目标关节缺失都会拒绝任务。
- 非必需缺失关节会在输出元数据中保留，例如源马尾和目标尾巴。
- 输出前执行 FK 播放检查、关节角度范围检查、根运动检查和接地脚滑动检查。
- 原始动作保留在原资源中；转换结果记录源动作、源/目标骨架、映射、缩放和校验报告。
- 输出 JSON 采用临时文件和原子替换；校验失败不写结果，已有资源默认绝不覆盖。

## 快速运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
anim-pipeline init-samples --root data
anim-pipeline run-batch --root data --batch data/batches/sample_walk_transfer.json
pytest
```

不安装包时可在项目根目录使用：

```bash
PYTHONPATH=src python -m anim_pipeline.cli init-samples --root data
PYTHONPATH=src python -m anim_pipeline.cli run-batch --root data --batch data/batches/sample_walk_transfer.json
```

## 数据约定

- `resources/skeletons/*.json`：层级、静息平移/旋转和关节范围。
- `resources/animations/*.json`：逐帧根部位移增量和关节局部旋转。
- `mappings/*.json`：源/目标关节对、必需关节和缩放模式。
- `batches/*.json`：一次处理中的资源路径和输出路径。
- `batches/runs/*.json`：批次执行报告；失败时列出错误且不生成目标动画。
