# ReKep 项目二｜从已部署环境到可投递项目档案

**项目名称**：工业物流与零售场景移动操作机器人关键点约束任务执行方案构建  
**角色定位**：仿真与任务建模负责人  
**目标输出**：个人代码仓库、两类场景任务配置与仿真证据、批量评测、故障分类、复用成本对照、可讲述的项目报告。  
**适用条件**：你已经在 Ubuntu 云服务器上配置好 ReKep / OmniGibson / Isaac Sim；使用现有 Conda 环境，不重装。  
**当前状态**：操作手册与脚本是可执行的工作框架；工业/零售场景资产、任务约束及成功率仍须在你的服务器验证。

## 0. 开始之前：最重要的 5 条约定

1. 所有命令都在**当前已有的 ReKep 仓库根目录**执行。文中的 `REKEP_ROOT` 是这一路径；`KIT` 是解压本交付包的位置。路径不要猜，先用 `pwd` 检查。
2. **不要**再次安装 Isaac Sim、改变 CUDA、覆盖原始 `main.py` 或直接把教学 Notebook 传 GitHub。先采集可复现证据，再在独立分支开发。
3. 官方 `python main.py --use_cached_query` 是笔筒任务；`--apply_disturbance` 是同一任务中的 **三阶段预设扰动组合**，不能说成工业物流/零售结果。官方缓存约束不能不经重生成直接用于不同物体。
4. **一个 Episode 必须保留**：场景 ID、初始状态 ID、随机种子、版本、原始命令、退出码、日志、视频及人工/程序明确验证的任务成功标记。`Video saved` 或 `Optimization terminated successfully` 均不是最终任务成功判据。
5. **12 布局 / 8 状态 / 87% / 17 类 / 40%** 是简历目标而非预填事实；必须保留有效配置、原始测试与统计脚本。不存在的结果宁可留空。

### 0.1 你已经提供的材料和它们在本项目中的位置

- `main.ipynb`：ReKep 的约束、IK、Subgoal、Path 与分阶段执行；对应 `source_audit/` 与基线分析。
- `main2_后退前进拿笔放进去.ipynb`：Fetch 底盘前后控制以及移动后重新获取 `rgb/points/seg`；对应移动链路验证，不自动等于自主导航。
- `rekep_config.ipynb`：Fetch 控制器、RGB-D 和操作空间控制参数参考。
- `关键点生成模块.zip`：关键点聚类和 `rgb.pkl / points.pkl / mask.pkl`；用于离线验证，不要把其中未经审查的 pickle 发给陌生人或直接放公开仓库。
- 历史日志出现 `OSC pose not reached`、`Target position is out of bounds`，可以作为**有待复验**的失败案例。
- 历史 Notebook 的 `API KEY` 需要撤销轮换，不可纳入公开代码或交付包。

### 0.2 目标产物与四条职责的映射

| 简历职责 | 工程落点 | 必须留存的证据 |
|---|---|---|
| 多阶段关键点约束建模 | 两份 `task_spec` + 关键点定义 + 阶段约束及通过条件 | `task_modeling.md`、约束函数/元数据、关键点图 |
| 移动抓取链路闭环 | 底盘位姿、RGB-D、目标/末端位姿、IK、Subgoal/Path、夹爪控制与结果检查 | 移动前后状态 JSON、Episode 日志、视频 |
| 多布局与失败定位 | 场景矩阵、相同判据的运行及失败分类 | `episodes.csv`、`failure_cases.csv`、代表视频 |
| 跨场景任务复用 | 公共 Stage Schema、工业/零售两个配置、同一 Adapter | 代码 Diff、测试结果、工时/代码修改量表 |

---

## 1. 第一次操作：环境与源码快照（约 10 分钟）

### 1.1 进入正确目录

在云服务器终端执行：

```bash
conda activate omnigibson
find "$HOME" -maxdepth 5 -type f -name main.py 2>/dev/null | grep -i rekep
```

找到路径后，手动进入其父目录；例如输出 `/home/ubuntu/Rekep/main.py`，则：

```bash
cd /home/ubuntu/Rekep
pwd
ls main.py environment.py keypoint_proposal.py ik_solver.py subgoal_solver.py path_solver.py
```

**验收**：以上 6 个文件存在。若缺文件，**停止本步骤并发送 `pwd` / `ls` 原样输出**，不要重新 clone。

### 1.2 把本交付包复制到服务器

在你的电脑上下载压缩包并上传到服务器，例如使用服务器图形桌面的文件管理器，放到 `~/Downloads/ReKep_Project2_Kit.zip`。随后：

```bash
unzip -q ~/Downloads/ReKep_Project2_Kit.zip -d ~/Downloads
export KIT="$HOME/Downloads/ReKep_Project2_Kit"
export REKEP_ROOT="$PWD"
mkdir -p "$REKEP_ROOT/project2_evidence"
bash "$KIT/scripts/00_preflight.sh" "$REKEP_ROOT" "$REKEP_ROOT/project2_evidence"
```

**保存**：`project2_evidence/00_environment/` 下版本信息、配置文件名和环境快照。脚本不改源码、不导出 API Key。

**检查**：

```bash
ls -lh project2_evidence/00_environment/
cat project2_evidence/00_environment/core_files.txt
```

**如果 `unzip` 提示不存在**：使用文件管理器确认 ZIP 的实际路径，替换上面路径即可。

### 1.3 创建自己的独立 Git 仓库（不要污染上游）

```bash
mkdir -p "$HOME/ReKep-Mobile-Manipulation"
cd "$HOME/ReKep-Mobile-Manipulation"
cp -r "$KIT/scripts" "$KIT/templates" "$KIT/docs" .
cp "$KIT/START_HERE.md" README.md
cp "$KIT/.gitignore" .gitignore
# 先只建本地仓库，不推送含凭据的历史 Notebook
git init
git add README.md scripts templates docs .gitignore
git commit -m "Initialize project2 workflow and evaluation protocol" || true
cd "$REKEP_ROOT"
```

**保存**：个人仓库的首个 commit ID 与 ReKep 上游 commit ID；前者只存你的工作文件，后者只作为依赖来源。

---

## 2. 第二步：跑官方基线（不用新 API 调用）

### 2.1 运行缓存查询 Pen-in-Holder

```bash
cd "$REKEP_ROOT"
conda activate omnigibson
bash "$KIT/scripts/01_run_rekep.sh" "$REKEP_ROOT" "$REKEP_ROOT/project2_evidence" baseline
```

脚本真正运行的命令是 `python main.py --use_cached_query`，捕获退出码、标准输出/错误、执行时间、新生成的视频清单。**第一轮不加 `--visualize`**，因为 Open3D 可能卡在窗口交互。

**保存**：

```text
project2_evidence/01_baseline/<run_id>/command.txt
project2_evidence/01_baseline/<run_id>/run.log
project2_evidence/01_baseline/<run_id>/exit_code.txt
project2_evidence/01_baseline/<run_id>/run_manifest.json
project2_evidence/01_baseline/<run_id>/new_videos.txt
```

运行后：

```bash
find project2_evidence/01_baseline -name run.log -print
find project2_evidence/01_baseline -name exit_code.txt -exec sh -c 'echo "$1"; cat "$1"' _ {} \;
ls -lh videos/ | tail -n 8
```

**判断**：退出码 0 + 有视频 = 程序执行证据；再**打开视频**检查是否实际完成目标。把 `success` 标为 `1 / 0 / 空缺`，不可通过优化器打印自动当作成功。

### 2.2 自动提取日志数值

```bash
python "$KIT/scripts/02_parse_log.py" \
  --log "$(find project2_evidence/01_baseline -name run.log | sort | tail -n 1)" \
  --out project2_evidence/01_baseline/latest_parsed.json
cat project2_evidence/01_baseline/latest_parsed.json
```

只提取实际日志里出现的 `stage`、`ik_feasible`、`solve_time`、`path_violation`、`OSC pose not reached`、`out of bounds`、视频路径。若没有字段则保留空值，不推断成功。

### 2.3 若发生错误，按这个顺序处理

| 现象 | 处理方式 | 本轮应保存 |
|---|---|---|
| 缺少 `main.py` | 核对路径；不重装 | `pwd`、`ls` |
| `botocore/httpchecksum` 导入错误 | 先停下，提供 `python -m pip show boto3 botocore s3transfer` 和报错尾部；暂不盲目升级 | 原始 `run.log` |
| `Open3D/GLX` 窗口错误 | 先用无 `--visualize` 命令，保留渲染和窗口报错 | 警告片段 |
| `OSC pose not reached` | 记录 stage、目标与实际末端误差；视频确认真实结果 | 原始日志 + 视频 |
| `Video saved` 但任务失败 | 如实标 `success=0`，归因不明时写 `unclassified` | Episode 记录 |
| 长时间未结束 | 记录运行持续时间、机器负载和最后 100 行日志；先定位死循环再处理 | 时间戳、尾部日志 |

---

## 3. 第三步：最省事地补移动操作证据

官方 `Main.perform_task()` 会 `env.reset()`；因此**不能**在外面先移动底盘再调用 `perform_task()`，否则移动可能被重置抹掉。

### 3.1 先运行独立底盘探针，不影响规划主流程

```bash
cd "$REKEP_ROOT"
python "$KIT/scripts/03_mobile_probe.py" \
  --repo "$REKEP_ROOT" \
  --out project2_evidence/02_mobile/mobile_probe.json \
  --steps 10 \
  --speed 0.10
```

该脚本在当前场景内获取移动前后 Fetch 基座位置和末端位置、相机输出键名，使用低幅度 `action[0]` 控制向前/后运动。**先确认实际控制器第 0 维确实对应底盘速度，且前方没有障碍物**。如果动作维度与示例不符，脚本会提示停止，不强行执行。探针只是“移动与观测接口检查”，不是自动导航。

**保存**：`02_mobile/mobile_probe.json`，比较 `base_before`、`base_after_forward`、`base_after_return`、`camera_keys`、末端位姿。若动作后基座无位移，则检查机器人控制器与动作索引，不能写成移动已验证。

### 3.2 再使用你已有的移动版 Notebook 做一次完整执行

1. 在本地保留 `main2_后退前进拿笔放进去.ipynb` 原件，不公开上传。
2. 创建无明文密钥的运行副本；确认运行顺序：**初始化 → 底盘后退/前进 → 重新获取 RGB-D → 关键点生成/加载约束 → Subgoal/Path → 执行**。
3. 如重置环境发生在移动之后，重新调整函数位置，保证规划使用**移动后的观测和机器人坐标变换**。
4. 记录移动前/后基座、相机与末端位姿，以及关键点坐标。`get_keypoint_positions()` 和 `get_cam_obs()` 读取的坐标系必须核对。
5. 保存日志和视频，人工检查最终结果。出现 `IK feasible=True` 但 `OSC pose not reached` 时不要把它当作任务成功。

**交付**：`02_mobile/mobile_full.log`、`mobile_full.mp4`、`mobile_state_before_after.json`、`mobile_integration_notes.md`。

---

## 4. 第四步：先做统一任务建模，再做两个场景

这是简历第 1 条职责。只写两个案例并不等于统一建模，必须把共性写进同一个 `Stage Schema`。

### 4.1 五阶段接口定义

| Stage | 输入 | 关键约束（示例） | 验收条件 |
|---|---|---|---|
| S1 接近 | 底盘位姿、目标位置 | 目标相对底盘距离/朝向处于工作范围 | 基座到达安全可操作区域 |
| S2 预抓取 | RGB-D、目标关键点、末端状态 | 末端接近抓取区域；姿态对齐与碰撞可行 | 预抓取位姿可达且无碰撞 |
| S3 抓取 | 夹爪、物体与关节状态 | 抓取候选满足几何约束、夹爪闭合 | 物体被可靠夹持 |
| S4 转运 | 目标放置区、抓持物体 | 物体保持抓持；运动路径可行 | 到达目标上方且物体未掉落 |
| S5 放置 | 目标区域、物体与末端 | 放置位置/朝向误差小于阈值 | 物体释放后保持在目标区域 |

**区分**：ReKep 的 `subgoal/path` 约束主要面向机械臂操作；S1 底盘导航并非官方 Cached Pen 任务自动提供。移动操作需要额外的底盘控制/可达区域检查，不能仅靠“阶段名字”完成。

具体约束模板（针对真正的物体需验证轴、尺寸、坐标系）：

- 接近距离：`d_min <= ||p_base - p_obj|| <= d_max`。
- 末端到抓取点：`||p_eef - p_grasp|| <= eps_grasp`。
- 朝向：`angle(R_eef * a_tool, a_approach) <= eps_rot`。
- 放置对齐：`||p_obj_center - p_goal_center|| <= eps_place`。
- 旋转对齐：`angle(R_obj, R_goal) <= eps_yaw`。
- 安全：碰撞代价与 IK 不可行时禁止直接进入下一阶段。

将每一个关键点的**物体归属、语义、坐标系、估计来源、可见性与阶段作用**写在 `templates/keypoints_template.csv`。如果用了仿真真值辅助求解，必须标记 `oracle_or_observation=oracle`。

### 4.2 工业物流最小闭环

目标任务：Fetch 在物料区接近一个可被当前夹爪实际抓取的容器/小箱体，将其转运至放置区。**优先选择当前 OmniGibson 资产库中已存在且尺寸适合 Fetch 的箱体**；不要直接选择超出夹爪开口/负载的料箱。

操作步骤：

1. 用当前 OmniGibson 图形界面或现有场景导出工具确认真实物体 `category / model / name / scale`，检查夹爪开口、物体质量及碰撞尺寸。
2. 从 `configs/og_scene_file_pen.json` 另存为 `configs/og_scene_file_logistics.json`；根据**已安装资产**编辑 `objects_info.init_info` 和 `state.object_registry`，不要仅把 `pen_1` 文本替换为 `box_1`。
3. 配置 `task_spec_logistics.yaml`（见模板），明确源区域、抓取区域、目标位置和阶段成功条件。
4. 为真实场景生成新关键点与约束，或人工准备完整匹配的 `metadata.json` + `stageN_subgoal_constraints.txt` / `stageN_path_constraints.txt`；**不可沿用笔筒任务的坐标和关键点索引**。
5. 第一次只跑一个固定位置，核对模型加载、关键点和目标区域；再启用底盘接近。保存 `scene_file`、实际生效的约束、完整视频。
6. 成功后再变化场景参数，不要一次更换多个因素。

**验收**：至少有一个工业场景从实际物体状态到抓取/放置完成的 Episode，包含视频、成功判据和结果记录。

### 4.3 零售商超最小闭环

目标任务：Fetch 接近货架/桌面商品，将单件商品抓取并放入购物篮/容器。初期可用同一可通行场景内的货架台面与篮子，不强求真实大型商超地图。

操作步骤与工业场景相同，但新建 `og_scene_file_retail.json`、`task_spec_retail.yaml` 与对应约束。重点不同在于：商品尺寸/材质、抓取区域、遮挡、目标容器安全放置条件。保持 `Stage Schema` 与评测脚本不变。

**验收**：零售和工业分别运行一个有效 Episode，并能指出**哪些 Python 函数完全相同、哪些只改配置/参数/约束**。

### 4.4 场景和约束程序检查（启动新任务前必须做）

工业/零售场景文件准备好以后，先**不启动 Isaac Sim**进行静态检查：

```bash
python "$KIT/scripts/08_inspect_scene.py" --scene configs/og_scene_file_logistics.json
python "$KIT/scripts/09_validate_program.py" --program_dir ./vlm_query/logistics
```

第一条展示 JSON 中实际存在的 object name / category / model / position；第二条检查关键点数、阶段数与约束文件结构。它们**不能验证物理碰撞，也不能证明约束语义正确**；通过后仍需一次真实仿真。

当你已经拥有工业场景文件和**与这个场景对应的**约束程序时，运行：

```bash
bash "$KIT/scripts/10_run_custom_task.sh" \
  "$REKEP_ROOT" "$REKEP_ROOT/project2_evidence" \
  logistics "$REKEP_ROOT/configs/og_scene_file_logistics.json" \
  "$REKEP_ROOT/vlm_query/logistics" \
  "pick the logistics box and place it inside the target area"
```

零售替换 `logistics`、场景文件、约束目录和任务句子即可。以上是**计划运行命令**，只有文件确实存在、资产匹配、场景和缓存约束正确时才可执行。否则先补对应文件；不要直接运行并假定它会自动创建资产或新的 VLM 约束。

### 4.5 新场景的工程阻塞条件

如果模型不存在、实例名不对、场景 JSON 无法加载、夹爪抓不到、没有目标关键点或约束不匹配，不要拿官方笔筒视频冒充新场景。保留阻塞信息；先交付已完成的模型与评测框架，再处理真实资产。此时如实标记 `scene_ready=false`。

---

## 5. 第五步：12 布局 × 8 状态的可复现配置矩阵

先生成**计划矩阵**，再把有效的布局变成真正可加载的场景。推荐 2 种任务 × 每种 6 布局 = 12 布局；每种布局配置 8 类目标状态，得到 96 个候选组合。

```bash
python "$KIT/scripts/04_make_matrix.py" \
  --out project2_evidence/04_scenarios/scenario_matrix.csv
```

| 维度 | 定义方式 | 例子 |
|---|---|---|
| 场景 | 2 类任务 | logistics / retail |
| 布局 | 每类 6 个，不是 6 次相同截图 | 左/中/右放置区、不同接近方向/障碍布局（需要保存真实配置差异） |
| 摆放状态 | 8 个物体初始状态 | 标准、X±偏移、Y±偏移、偏航±角、靠近边缘 |
| 重复次数 | 每个有效组合多个独立种子 | seed_00, seed_01, seed_02 |

`scenario_matrix.csv` 的初始字段是 `scene_ready=0`、`validated=0`，仅代表**设计方案**。只有生成了场景文件、实际完成重置并检查碰撞、可达性和物体稳定性，才能标记为 `validated=1`。

如已有可运行场景 JSON，可在安全范围内生成目标位置变体（不会新增资产）：

```bash
python "$KIT/scripts/05_scene_pose_variant.py" \
  --source configs/og_scene_file_pen.json \
  --object pen_1 --dx 0.02 --dy 0.00 --yaw_deg 0 \
  --out project2_evidence/04_scenarios/pen_shift_x02.json
```

以上命令**只为当前笔对象生成示例位姿变体**，不是工业或零售的测试数据；真正场景需要替换为对应已存在的对象实例名，并检查物理可行性。

---

## 6. 第六步：实验运行、标注和统计

### 6.1 实验协议（不能中途改口径）

- **实验单位**：一个重置后的完整任务 Episode。
- **成功**：最终物体属于正确目标、位姿误差在预定义阈值内、物体释放后稳定、无安全违规，并且 S1-S5 都满足；碰撞/超时/掉落计失败。
- **同一组比较**：同一场景/初始状态/随机种子/超时阈值。不要把简单场景混进改进组。
- **环境记录**：ReKep commit、Conda 包版本、机器人控制参数、场景 JSON、约束文件 Hash、随机种子。
- **数据类型**：成功必须经最终状态检查确认；规划器成功、IK 可行、视频生成是**中间指标**，不能替代任务成功。
- **禁止误标**：场景未运行、日志缺失、无法判断成功，`success` 留空而不是自动填 1。

### 6.2 Episode CSV 规范

填 `templates/episodes_template.csv` 的各列：`run_id,scenario,layout_id,placement_id,seed,method,scene_file,program_dir,success,...`。

可复制一份开始逐条填：

```bash
mkdir -p project2_evidence/05_evaluation
cp "$KIT/templates/episodes_template.csv" project2_evidence/05_evaluation/episodes.csv
```

一次 Episode 完成后：手工/程序检查最终状态；填 `success=1` 或 `0`；如果失败填首要 `failure_code`、`failure_stage`、`video_path`。如已有真实世界坐标，可另存 `state_trace.csv`（时间、base、eef、target、grasp、constraints）。

### 6.3 指标定义及自动计算

```bash
python "$KIT/scripts/06_aggregate.py" \
 --input project2_evidence/05_evaluation/episodes.csv \
 --out project2_evidence/05_evaluation/summary.json
```

脚本会**跳过未标注**的 Episode，并输出样本量、成功数、整体成功率、工业/零售分组成功率、各失败码的数量，不会自动凑 87%。

`overall_success_rate = valid_success / valid_total`。如果结果接近 87%，才在简历写实际数值，并同时说明 `n` 与场景分布。例如 `167 / 192 = 86.98%` 是数学演示，**不是本项目当前实验结果**。

### 6.4 17 类问题如何形成

`templates/failure_taxonomy.csv` 给出 17 **候选**故障码，覆盖感知、任务约束、规划、控制及阶段管理。不能因为模板有 17 条，就说“分析了 17 类实际问题”。必须每类至少有有效日志/视频或重现场景；无证据的代码保持 `observed=0`。失败现象和根因要分开，不能把 `OSC pose not reached` 直接等同于 IK 失败。

### 6.5 开发成本降低 40% 如何测

建议只选择一个可复核口径：**新增任务的实际开发工时（排除等待 GPU、下载、休息）**。

- **独立模式**：不用通用 Stage Schema，从单任务实现新增逻辑，记录编码 + 调试分钟数。
- **复用模式**：沿用 Stage Schema/Adapter，仅新增配置、少量约束，记录同等工作范围的分钟数。
- `reduction=(T_independent-T_reuse)/T_independent*100%`。
- 用 `templates/reuse_cost.csv` 记录任务、开始/结束、范围、计时方式。只有做了可比实验并得到约 40% 时，才写这个指标。若只量化了文件行数，则写“适配代码量降低”，不能换称开发时间降低。

---

## 7. 第七步：个人 GitHub 仓库与材料打包

仓库保持“薄适配层”：上游 ReKep 以 commit 引用，个人仓库包含 Stage Schema、两场景任务配置、运行包装脚本、评测脚本、统计结果和报告。不要把大模型权重、OmniGibson 资产、旧 Notebook 中的密钥、私有服务配置推到公开仓库。

建议 `README` 首屏按以下顺序：项目问题 → 架构 → 运行命令 → 工业/零售各一个 Demo → 指标表（带 n）→ 技术贡献 → 失败案例 → 可复现性 / 局限性。

**发给我的数据包**最好包含：

```text
project2_submission/
  environment_summary.txt       # 版本信息，不含 token
  source_commit.txt             # 上游和个人仓库 commit
  task_spec_logistics.yaml
  task_spec_retail.yaml
  actual_scene_configs/         # 两类已验证的 JSON（非大体积资产）
  actual_constraints/           # 每场景 metadata + stage 约束
  baseline_run.log              # 清理 token 后
  mobile_probe.json
  episodes.csv                  # 每一行有真实 run_id
  scenario_matrix.csv           # 标明 validated/scene_ready
  failure_cases.csv             # 可对照的日志和视频
  reuse_cost.csv                # 复用工时测量
  results_summary.json
  representative_videos/        # 每场景成功/失败各至少一个
  screenshots/                  # 关键点、规划/放置状态
```

上传前检查：

```bash
# 只在自己的项目中扫描潜在密钥，不要将匹配内容打印发给我
python "$KIT/scripts/07_secret_scan.py" "$HOME/ReKep-Mobile-Manipulation"
```

`07_secret_scan.py` 返回可疑**文件路径和行号**，不打印密钥内容。绝不要直接把 `.env` 或含 key 的 Notebook 上传。

---

## 8. 第八步：你交数据后，我将如何生成最终面试档案

最终报告必须有：

1. **1 页项目摘要**：业务背景、为何选 ReKep、个人工作边界、两个场景任务。
2. **系统架构与代码调用链**：RGB-D → 关键点 → 约束 → IK/子目标/路径 → 底盘/机械臂 → 状态反馈。
3. **任务建模技术细节**：每场景 S1-S5、关键点坐标系、约束表达、阈值与阶段切换条件。
4. **仿真部署与接口**：机器人型号、控制器、状态查询、相机观测、版本及操作命令。
5. **移动闭环案例**：移动前后观察与关键点更新、世界/机器人/相机坐标转换和重规划调用。
6. **两类场景复用**：共用组件、场景差异、代码 Diff、复用工时统计。
7. **实验协议和结果**：12 布局、8 类状态的有效组合、样本数、成功率及区间估计。
8. **失效分析**：诊断图、日志证据、17 类候选中真正观察到的类别、对应优化方向。
9. **方法局限**：仿真真值依赖、真实感知缺失、资源与样本量限制。
10. **面试准备**：2 分钟/5 分钟/15 分钟讲述；关键追问题与对应代码位置。

### 8.1 面试故事固定框架

**业务问题**：两个行业的移动抓取具有相同阶段骨架，但对象几何、抓取区域、放置目标及可达性条件不同。  
**困难**：感知关键点、约束、底盘位姿与末端执行状态需要保持一致；阶段切换、IK 与控制会产生不同类型失败。  
**工程方案**：通过统一 Stage Schema 与任务配置表达不同任务，由 ReKep 求解抓取放置子目标；底盘控制和状态更新提供移动操作前后的可执行条件；结构化记录诊断信息。  
**实验设计**：两种任务、不同场景布局与摆放、相同成功判据和日志口径、分组成功率/失败码/工时对照。  
**结论**：只能根据最终真实统计结果下结论，给出有效样本量、局限性和下一步技术改进方向。

### 8.2 最小的学习路线（边做边补）

| 你要回答的问题 | 必须掌握的知识 | 应关联的代码 |
|---|---|---|
| 为什么关键点能表达任务？ | 3D 坐标、物体局部/世界坐标、相对约束 | `keypoint_proposal.py`, `constraint_generation.py` |
| VLM 到底做什么？ | 约束生成与缓存、避免把模型说成在线动作策略 | `vlm_query/`, `ConstraintGenerator` |
| IK feasible 为何仍会失败？ | 运动学 vs 轨迹执行、OSC 误差、碰撞和抓持 | `ik_solver.py`, `environment.py` |
| `Subgoal` 与 `Path` 区别？ | 阶段目标位姿优化 vs 中间轨迹优化 | `subgoal_solver.py`, `path_solver.py` |
| 移动后为什么要更新观测？ | 基座变化、坐标变换、可达空间变化 | 移动 Notebook、`get_cam_obs()` |
| 如何证明跨场景复用？ | 抽象复用边界、相同调度接口、差异配置 | `task_spec_*.yaml`, 适配层 Git Diff |
| 87% 有没有统计意义？ | 分母、种子、置信区间、是否全组合测试 | `episodes.csv`, `summary.json` |
| 40% 是怎么测的？ | 相同任务范围、工时定义、对照设计 | `reuse_cost.csv` |

---

## 9. 最短执行路径：一轮只做一个任务

**今天只做 3 个动作**：

```bash
# A. 在已经安装好的 ReKep 根目录
conda activate omnigibson
cd /你的/实际/ReKep/目录
export REKEP_ROOT="$PWD"
export KIT="$HOME/Downloads/ReKep_Project2_Kit"

# B. 生成环境快照
bash "$KIT/scripts/00_preflight.sh" "$REKEP_ROOT" "$REKEP_ROOT/project2_evidence"

# C. 跑一遍已有缓存任务并留证
bash "$KIT/scripts/01_run_rekep.sh" "$REKEP_ROOT" "$REKEP_ROOT/project2_evidence" baseline
```

然后把 `00_environment`、`01_baseline` 的版本记录、最后 100 行日志和一个实际视频发来。第一轮**不需要**搞 12 布局、8 状态，也不需要再发整套教程截图。确认基线后再做移动探针和两个新场景。

### 你每轮反馈统一用这个格式

```text
当前阶段：阶段 1/2/3/... 
执行命令：...
返回码：0 / 其他
本轮生成的文件：...
运行视频结果：成功 / 失败 / 无法判断
观察到的报错：...
我发来的附件：...
```

---

## 10. 参考来源与实现边界

- ReKep 官方代码及运行入口：`https://github.com/huangwl18/ReKep`；其 `main.py` 有缓存任务、三阶段扰动、状态更新和阶段回退。官方 Demo **不是工业/零售两场景的现成实现**。
- ReKep 官方 README：公开仿真代码缺少真实世界关键点跟踪、Mask 跟踪和 SDF 重建的完整感知链；仿真代码有使用真值状态的环节，报告中必须区分。
- 用户上传的 Notebook 与历史日志：用于制定项目开发接口和可分析失败项；所有最终运行结果以云服务器实际留存的 Episode 为准。
