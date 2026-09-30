# 动画资源加工系统

这是一个从零搭建的 Python 标准库项目，第一批处理能力是跨骨架动作转移。资源、骨架映射和处理批次独立存储，后续压缩、格式转换或导出器可以复用现有资源模型、原子写入和批处理记录。

## 目录结构

- `anim/math3d.py`：向量、四元数和绑定轴运算。
- `anim/models.py`：骨架、动作、映射、批次 JSON 模型与原子写入。
- `anim/retarget.py`：绑定姿态相对旋转转移、映射骨长缩放、根部位移和正向运动。
- `anim/contact.py`：接触区间锚定、根位移补偿、腿部 IK 与未满足接地残差。
- `anim/merge.py`：多片段重定向、局部修订、片段连续缝合和来源/映射版本追踪。
- `anim/merge_validation.py`：接触滑动、离地脚冻结和片段根位移间隙检查。
- `anim/validation.py`：脚部世界位置和关节角度范围检查。
- `anim/pipeline.py`：批处理、失败隔离、不覆盖已有资源和运行记录。
- `anim/playback.py`：在目标骨架上评估每帧关节世界坐标。
- `data/resources/`：两套骨架、源动作和关节映射。
- `data/batches/`：独立保存的处理批次。

## 转移方法

动作不是简单改关节名称：

1. 读取每个关节的父关节、骨长偏移和 `bind_axes` 初始姿态。
2. 计算源关节相对自身绑定姿态的全局旋转变化。
3. 将相同旋转变化应用到目标关节的全局绑定姿态，再转换为目标父关节下的本地四元数。
4. 根据映射关节总骨长计算比例，缩放根部位移。
5. 显式保留源缺失关节和目标独有关节，不把它们伪装成成功映射。

## 动作验证

批次写出前执行两类检查：

- 对目标动作做正向运动，比较双脚世界位置与源脚点按比例、根部位移重放后的位置。
- 将目标本地四元数转换为欧拉角，检查每个配置的关节范围。

任一检查失败时，该任务不会写动作资源。默认使用临时文件和原子替换；已有文件默认拒绝覆盖。每次批次都会在 `records/` 下写带时间戳的记录，保留原始资源路径、转换资源路径、验证结果和缺失关节信息。

## 使用

```bash
python -m anim.cli run-batch \
  --resources data/resources \
  --batch data/batches/continuous_demo_batch.json \
  --output /tmp/animation-output

python -m anim.cli preview \
  --skeleton data/resources/skeletons/simple_target.json \
  --motion /tmp/animation-output/motions/walk_simple_target.json
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 示例数据

`simple_target` 的骨长约为源骨架的 1.25 倍，并且所有绑定轴相对源骨架旋转 90 度。目标还包含源没有的 `back_gear`，源包含目标没有的 `head_top`，映射中分别列为目标独有和源缺失。源动作包含逐帧根部位移，输出动作记录目标骨架名、缩放比例、原始动作引用和完整目标关节四元数。

## 脚部接地与连续动作

动作资源可在 `contacts` 中为每只脚声明接触区间，例如：`{"l_foot": [{"start": 0, "end": 4}]}`。

- 接触首帧建立世界锚点；区间内接触中的脚不能沿地面滑动。
- 根位移先做整体补偿，髋、膝再通过受关节范围约束的腿部 IK 缩小接地误差。
- 未列入接触区间的帧不会被接地求解器固定，仍保留摆动轨迹。
- 当腿长、可达距离或关节范围阻止完全接地时，动作不会被强行拉坏；输出保留可播放姿态，并在 `unmet_contacts` 中记录帧、脚、锚点、实际点和残差原因。
- 映射文件通过 `version` 标识版本。合并输出的 `provenance` 为每段保留原动作资源、映射名称/版本、帧范围和局部修订。
- 合并批处理使用 `kind: "merge_clips"`。`data/batches/continuous_demo_batch.json` 使用不同腿长、转向和上下台阶片段，核对根位移、接触误差和片段衔接。
