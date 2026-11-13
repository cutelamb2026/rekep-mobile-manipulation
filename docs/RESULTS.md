# 实验结果与证据边界

以下结论依据随项目打包的 `project2_evidence` 中 JSON 和 CSV 记录，不进行额外虚构评测。

## 26 次参数化实验

- 来源：[`observed_episodes.csv`](../project2_evidence/07_evaluation/observed_episodes.csv)
- 共 26 次；程序 `program_success=True` 为 **25/26 = 96.15%**。
- 程序成功且最终水平误差 ≤ 0.03 m 的为 **24/26 = 92.31%**。
- 失败任务：`logistics_L04_P01`。一项成功任务虽程序成功但超过 3 cm。
- 这是预先定义位置偏移的参数化实验，并不是 12 个各不相同的物理场景布局，也不等同于独立随机试验的成功率。
- 全部 26 项记录的 `process_exit_code=-11`；进程完整性问题仍未解决。

## 工业物流：独立 A/B 工位

来源：[`logistics_run02/result.json`](../project2_evidence/13_final_dual_station/logistics_run02/result.json)

- `status=finished`、`success=True`（程序自身成功判据）。
- `grasp_verified=True`、`transport_grasp_retained=True`。
- 两个移动航点均到达；底盘携载位移 **0.36579668494491 m**。
- B 区域移动后 RGB-D 视觉估计：[0.05112108, 0.30096239, 0.69012578] m。
- 该估计与配置 B 参考位置之间的水平差约 **0.00147749 m**。这是仿真场景内部参考误差，不代表真实部署中的感知精度。
- `runtime_constraints_updated=True`、`release_called=True`。
- 最终水平误差 **0.03951236 m = 3.95 cm**，**不满足 3 cm 严格标准**。
- 视频路径在源服务器：`videos/2026-11-05-17-45-19.mp4`。本压缩包不包含视频，因此“最终物料已稳定落在 B 台面”需要用户再结合视频确认。
- 运行后出现 `PROCESS_EXIT_CODE=139`；不能据 `status=finished` 判定仿真程序能够正常清理退出。
- 工位约 48.86 cm 间距且具有独立碰撞几何，但没有完成全场景物理碰撞的完整认证；基于 AABB 的路径预筛查属于保守几何检查。

## 零售取货

来源：[`retail_v1/result.json`](../project2_evidence/09_vision_execution/retail_v1/result.json)

- `success=True`、`release_called=True`。
- 水平落位误差 **0.01820280 m**。
- 示例中的商品颜色 / 仿真分割等信息参与目标选择；**不宣称零样本真实 SKU 识别**。

## 方法限制

- 工业场景 RGB-D 识别依赖已知颜色与仿真点云；并非训练出的通用物体识别模型。
- 注册关键点仍使用仿真网格吸附/对象绑定；不能说彻底去除了 simulator privileged information。
- 没有完成真实实体机器人部署、正式避障导航性能评估或大规模独立统计验证。
- 先前设想的 17 类故障、40% 效率改进和 87% 成功率等**目标**不能当作实测结果发布。
