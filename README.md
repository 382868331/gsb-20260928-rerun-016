# 动画资源加工系统

这是一个从零搭建的 Python 标准库项目，第一批处理能力是跨骨架动作转移。资源、骨架映射和处理批次独立存储，后续压缩、格式转换或导出器可以复用现有资源模型、原子写入和批处理记录。

## 目录结构

- `anim/math3d.py`：向量、四元数和绑定轴运算。
- `anim/models.py`：骨架、动作、映射、批次 JSON 模型与原子写入。
- `anim/retarget.py`：绑定姿态相对旋转转移（含绑定轴坐标系共轭）、映射骨长缩放、根部位移和正向运动。
- `anim/contacts.py`：多末端接触区间解析（脚/手）、重叠接触共享根修正、优先级让步、移动目标关键帧、进出混合与两骨 IK。
- `anim/plans.py`：接触约束方案的独立重算、并排比较排序与择优保存。
- `anim/merge.py`：多片段拼接、逐段根偏移/转向放置、接缝根位移与接触滑步核对。
- `anim/validation.py`：脚部世界位置、关节角度范围和接触滑步检查。
- `anim/pipeline.py`：批处理（转移与合并任务）、失败隔离、不覆盖已有资源和运行记录。
- `anim/playback.py`：在目标骨架上评估每帧关节世界坐标。
- `data/resources/`：多套骨架（含不同腿长的高个目标）、直行/转向/上下台阶源动作和带版本的关节映射。
- `data/batches/`：独立保存的处理批次（含手扶栏杆上下台阶的多目标接触任务）。
- `data/contact_plans/`：同一段动作的多套接触约束方案，用于并排比较与择优保存。

## 转移方法

动作不是简单改关节名称：

1. 读取每个关节的父关节、骨长偏移和 `bind_axes` 初始姿态。
2. 计算源关节相对自身绑定姿态的全局旋转变化。
3. 旋转变化经绑定轴共轭转换到目标坐标系后应用到目标关节的全局绑定姿态，再转换为目标父关节下的本地四元数。
4. 根据映射关节总骨长计算比例，根位移先旋转到目标绑定轴再缩放。
5. 显式保留源缺失关节和目标独有关节，不把它们伪装成成功映射。

## 多目标接触约束

动作资源在顶层 `contacts` 列表中为任意两骨末端（`*_foot`、`*_hand`）声明接触区间，支持同一帧多个重叠目标（例如一只脚踩台阶、另一只脚尚未离地、同时一只手扶住栏杆）。每个区间可包含：

- `start`/`end` 与静态地面 `ground_y`，或 `target_keyframes`（逐帧世界目标点）描述随支撑物移动的栏杆、自动扶梯/升降平台；关键帧之间线性插值。
- `priority`：数值越小优先级越高。同一帧的接触共享根部位移，先满足高优先级目标；低优先级目标若会把更强目标拉出可达范围，则被明确让步（`conceded`），而不是被静默丢弃或伪装成功。
- `blend_in`/`blend_out`：进入/退出接触的过渡帧数，末端从原动作平滑过渡到约束目标，再平滑释放；过渡帧权重从 0 递增到 1，只有满权重帧参与滑动检查。
- 区间之外（以及接触权重为 0）的帧完全保留重定向后的原动作，自由手臂、摆动腿不会被固定。

求解方式：

1. 按优先级一次性汇总共享根修正（每个接触沿基座到目标方向做最小位移，比例不同的角色仍保持自然姿态），再对所有腿/手臂两骨链（髋-膝-踝、肩-肘-腕）迭代 IK 直至收敛，避免后求解的肢体把已钉住的末端拉走。
2. 关节范围冲突时关节被截断到限制内，剩余残差以 `unmet_contact` 告警记录（含锚点、残差、根修正量、冲突轴、让步对象）。无法同时满足时各目标分别报告误差与状态（`satisfied`/`conceded`/`unmet`），不会把所有目标都标为成功。
3. 移动支撑的脚逐帧跟随关键帧；验证对满权重帧逐帧比较世界位置（5 mm 容差），让步帧不计为滑动失败。
4. 输出文档的 `grounding` 段保存锚点、移动关键帧、混合窗口、优先级、每区间状态/最大残差/让步对象。

## 约束方案比较与保存后核对

同一段动作可以准备多套接触约束方案（不同优先级组合、静态高度或移动目标），通过 CLI 从全新重定向结果分别重算并排序：

```bash
python -m anim.cli compare-plans \\
  --resources data/resources \\
  --source-skeleton skeletons/simple_source.json \\
  --target-skeleton skeletons/simple_target.json \\
  --source-motion motions/railing_step_source.json \\
  --mapping mappings/simple_source_to_simple_target.json \\
  --plans data/contact_plans/railing_step_plans.json \\
  --save-selected /tmp/railing_selected.json
```

排名依次比较滑动违规、未满足区间、让步区间、最大接触误差和让步帧数；输出每个方案中每个接触的状态、残差和让步对象。批处理任务也可以直接内嵌 `contact_plan`。保存后系统会对写出的文档重新加载并核对（`saved_verification`），覆盖直行转向、上下台阶以及移动支撑物片段；合并片段的转向/根偏移会同步变换接触锚点与移动关键帧。

## 多片段合并

`merge` 批次任务将已经接地的目标动作按帧拼接：

- 每个片段独立保留源资源、`mapping_version` 和 `local_revisions`，不会因为拼接丢失出处或局部修订。
- 每个片段可以给出 `root_yaw`（转向）和 `root_offset`（上/下台阶等根位移）；片段按上一段结束根位置锚定放置。
- 每个接缝输出根位移间隙、双脚跨缝位移，以及跨越接缝仍处于接触的脚的滑动量。
- 帧率不一致或骨架不匹配直接报错；接触滑动以 5 mm 容差检查。
- 转向和根偏移放置会把静态锚点与移动目标关键帧一同变换到合并后的世界坐标。

## 动作验证

批次写出前执行两类检查：

- 对目标动作做正向运动，比较双脚世界位置与源脚点按比例、根部位移重放后的位置。
- 将目标本地四元数转换为欧拉角，检查每个配置的关节范围。
- 对声明的接触区间做逐帧世界位置比较，接触中的脚不能滑动。

关节范围或接触滑动检查失败时，该任务不会写动作资源；可解释的 `unmet_contact` 告警（关节范围冲突或向高优先级目标让步后的剩余误差）不会阻止写出，但每个接触都会单独标注状态，不会把未满足目标记为成功。保存后会重新加载写出的文档并核对（转向、上下台阶、移动支撑物均在覆盖范围内）。默认使用临时文件和原子替换；已有文件默认拒绝覆盖。每次批次都会在 `records/` 下写带时间戳的记录，保留原始资源路径、转换资源路径、映射版本、局部修订、接地结果、保存后核对、接缝指标和缺失关节信息。

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
