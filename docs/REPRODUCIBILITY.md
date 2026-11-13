# 复现环境与路径

## 项目包包含什么

- 项目自行编写的扩展与任务集成脚本。
- `configs/og_scene_file_dual_station_v2.json`：序列化双实体工位场景。
- 关键结果 JSON、26 次参数化评测 CSV。

## 包外依赖

- 兼容 NVIDIA GPU 的 Linux 仿真环境。
- ReKep 上游仓库及其完整依赖。
- OmniGibson 4.1.0 / 与之兼容的 Isaac Sim、数据资产。
- 原项目的 `main.py`、`environment.py`、`configs/config.yaml` 等；本仓库不打包这些上游项目文件。
- 原始运行视频并未随包上传。

## 在原 ReKep 工作目录中恢复项目脚本

以下是结构参考，不是经过新机器实测的一键安装：

```bash
# 假设已经在独立的 Linux GPU 环境中配置好 ReKep/OmniGibson
cd /path/to/Rekep
# 从本仓库复制 project2_tools/ 与 configs/og_scene_file_dual_station_v2.json
# 仔细核对 configs/config.yaml、对象资源、坐标边界及配置版本
conda activate omnigibson
python -X faulthandler -u project2_tools/25_final_dual_station_logistics.py
```

**注意：** 最终执行器期望之前的路线安全预检查 JSON 存在于 `project2_evidence/12_dual_station/route_preflight/report.json`；本次压缩包仅包含精选数据，并未导出该预检查文件。需要先在仿真端重做预检查或从原云服务器单独复制该文件后，才有条件尝试完整复现。此外，`Main` 的相机/配置/约束接口依赖原上游环境。**本仓库当前定位为工程展示与实验存档，不宣称脱离原云端环境即可直接运行。**

## Mac 用户

Mac 不需要安装 Isaac Sim。用 GitHub Desktop 维护仓库、写 README、查看代码、保存轻量结果；保持原云服务器用于必要的高性能仿真。

## 资料管理

未包含上游资产和源代码，避免混淆授权归属。个人仓库准备公开之前，仍应核对第三方项目许可、导出的场景 JSON 是否允许公开，以及文件中是否有隐私/凭据。
