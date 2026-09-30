# 动画资源加工系统

这是一个从零搭建的 Python 标准库项目，第一批处理能力是跨骨架动作转移。资源、骨架映射和处理批次独立存储，后续压缩、格式转换或导出器可以复用现有资源模型、原子写入和批处理记录。

## 目录结构

- `anim/math3d.py`：向量、四元数和绑定轴运算。
- `anim/models.py`：骨架、动作、映射、批次 JSON 模型与原子写入。
- `anim/retarget.py`：绑定姿态相对旋转转移（含绑定轴坐标系共轭）、映射骨长缩放、根部位移和正向运动。
- `anim/contacts.py`：接触区间解析、两骨腿部 IK 接地、水平/垂直根修正和关节限制冲突报告。
- `anim/merge.py`：多片段拼接、逐段根偏移/转向放置、接缝根位移与接触滑步核对。
- `anim/validation.py`：脚部世界位置、关节角度范围和接触滑步检查。
- `anim/pipeline.py`：批处理（转移与合并任务）、失败隔离、不覆盖已有资源和运行记录。
- `anim/playback.py`：在目标骨架上评估每帧关节世界坐标。
- `data/resources/`：多套骨架（含不同腿长的高个目标）、直行/转向/上下台阶源动作和带版本的关节映射。
- `data/batches/`：独立保存的处理批次。

## 转移方法

动作不是简单改关节名称：

1. 读取每个关节的父关节、骨长偏移和 `bind_axes` 初始姿态。
2. 计算源关节相对自身绑定姿态的全局旋转变化。
3. 旋转变化经绑定轴共轭转换到目标坐标系后应用到目标关节的全局绑定姿态，再转换为目标父关节下的本地四元数。
4. 根据映射关节总骨长计算比例，根位移先旋转到目标绑定轴再缩放。
5. 显式保留源缺失关节和目标独有关节，不把它们伪装成成功映射。

## 脚部接地

动作资源可以在顶层 `contacts` 列表中为任意 `*_foot` 关节声明接触区间（`start`/`end` 帧和该区间的 `ground_y`）：

- 接触中的脚在整段区间被钉在区间起点的水平锚点与指定地面高度上，通过两骨（髋/膝/踝）IK 求解；必要时沿水平方向或垂直方向最小化修正根位置，平地根高保持不变。
- 摆动帧完全不修改，离地阶段不会被固定。
- 当配置的关节范围与接地目标冲突时，越界关节被截断到限制内，剩余偏移以 `unmet_contact` 告警（含锚点、残差、根修正量和具体冲突轴）记录；动作仍被写出，不会被强行拉到关节范围之外。
- 输出文档的 `grounding` 段保存每个区间的锚点、最大残差和未满足帧；验证只把真正的接触滑动判为失败。

## 多片段合并

`merge` 批次任务将已经接地的目标动作按帧拼接：

- 每个片段独立保留源资源、`mapping_version` 和 `local_revisions`，不会因为拼接丢失出处或局部修订。
- 每个片段可以给出 `root_yaw`（转向）和 `root_offset`（上/下台阶等根位移）；片段按上一段结束根位置锚定放置。
- 每个接缝输出根位移间隙、双脚跨缝位移，以及跨越接缝仍处于接触的脚的滑动量。
- 帧率不一致或骨架不匹配直接报错；接触滑动以 5 mm 容差检查。

## 动作验证

批次写出前执行两类检查：

- 对目标动作做正向运动，比较双脚世界位置与源脚点按比例、根部位移重放后的位置。
- 将目标本地四元数转换为欧拉角，检查每个配置的关节范围。
- 对声明的接触区间做逐帧世界位置比较，接触中的脚不能滑动。

关节范围或接触滑动检查失败时，该任务不会写动作资源；可解释的 `unmet_contact` 告警（关节范围与接地冲突后的剩余误差）不会阻止写出。默认使用临时文件和原子替换；已有文件默认拒绝覆盖。每次批次都会在 `records/` 下写带时间戳的记录，保留原始资源路径、转换资源路径、映射版本、局部修订、接地结果、接缝指标和缺失关节信息。

## 使用

```bash
python -m anim.cli run-batch \
  --resources data/resources \
  --batch data/batches/demo_batch.json \
  --output /tmp/animation-output

python -m anim.cli run-batch \
  --resources data/resources \
  --batch data/batches/multi_segment_batch.json \
  --output /tmp/animation-multi

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
