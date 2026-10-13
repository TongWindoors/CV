# 房价预测：组员复现包

**第一次使用：下载 Release 中的 `HOUSE_PRICE_TEAM_FULL.zip`，完整解压，双击 `00_START.cmd`。**

面向 Windows 10/11 x64。无需 VS Code，无需管理员，无需之前的 V10/V11 文件夹。完整包包含原图、标签、固定划分、已有冻结特征、便携 Python 和基础依赖离线安装包。不要在 ZIP 预览窗口内运行。

默认流程重新训练四个版本的回归器，验证后生成四份预测。它复用已提取的视觉特征，不进行视觉骨干微调。GPU 从原图重新编码是另一个明确入口。

## 先认识四个版本

这是**报告里的新编号**，不是把历史文件名 V1/V2/V3 原样重发。Baseline 是“本研究的预训练特征线性基线”，不是课程官方 baseline。

| 编号 | 模型与输入 | 这一步研究的问题 | 同条件 OOF MSE |
|---|---|---|---:|
| Baseline | 冻结 DINOv2 with registers Large，336 输入，CLS＋有效图像区域 patch 均值；Ridge | 简单线性读出能利用多少信息？ | 81,845.2503 |
| V1 | 完全相同的单 DINO 特征；RBF 核岭回归 KRR | 非线性回归是否优于线性？ | 77,861.0584 |
| V2 | 三个冻结 DINO 表征的固定距离融合；相同 KRR | 多骨干/分辨率表征是否互补？ | 76,311.7698 |
| V3 | V2 的 DINO 融合＋SigLIP2 Large 全图视觉表征；相同 KRR | 不同预训练目标的视觉表征是否互补？ | 68,046.7997 |

以上是 2026-10-13 本机真实重新拟合得到的结果，使用同一 5,118 张图片、同一三个按重复图片组划分的折；每折拟合 3,412 张，评估 1,706 张。V3 比 Baseline 的同折 MSE 低约 16.86%。这些不是 Kaggle 分数，且划分与超参数曾用于历史研究，不能当作新的盲测或因果证明。完整逐图预测和日志在 `evidence/current_reproduction/`。

V1→V2 同时引入不同 DINO 骨干与分辨率，只能归因于这一整套表征融合；不能据此单独声称“大模型”或“高分辨率”一定更好。

早期 RGB／six 是历史探索，未作为这四个报告版本：当前最完整、可对齐的证据来自上述冻结特征路线。其旧指标不与这里的同折主表混写。

## 怎么运行

1. 将完整包解压到普通文件夹，例如 `D:\house_price_team`，至少留出约 4 GB 可用空间用于环境和运行结果。
2. 双击 `00_START.cmd`。若电脑没有适用的 Python，程序使用包内便携 Python；在包内建立独立 `.venv`。
3. 首先校验依赖、数据和特征，然后依次运行四个方法的三折比较，再分别使用全部 8,000 张训练图片的标签重拟合，预测 3,000 张测试图片。
4. 完成后打开 `runs/本次编号/REPORT.html`。把根目录的 `TEAM_RESULTS.zip` 发回即可，里面包括配置、运行日志、逐折分数、逐图预测、提交 CSV 和代码快照。

默认在已有特征上计算，不需要 NVIDIA 显卡，也不会显示深度学习 epoch。Ridge/KRR 的回归权重通过线性代数求解，解出来即完成，不需要人为重复训练九小时。CPU 与显卡提取特征的角色不同。

| 入口 | 用途 |
|---|---|
| `00_START.cmd` | 四个版本完整复现＋四份最终预测 |
| `10_BASELINE.cmd` | 只跑 Baseline |
| `11_V1.cmd` | 只跑 V1 |
| `12_V2.cmd` | 只跑 V2，纯视觉预训练路线 |
| `13_V3.cmd` | 只跑 V3，含图文预训练来源的 SigLIP2 视觉分支 |
| `01_GPU_REEXTRACT.cmd` | 用 NVIDIA GPU 从原图重新提取四类特征 |
| `03_GPU_COMPARE.cmd` | GPU 重提完整且校验通过后，使用新特征重跑四版本 |
| `STOP.cmd` | 请求在安全检查点停止；线性代数求解期间可能要等待当前操作完成 |
| `COLLECT_RESULTS.cmd` | 重新收集最近一次回归运行的结果，不重新训练 |

每次训练建立新的 `runs/` 子目录。旧结果包移入 `result_history/`，不会冒充本次成功。主动停止后，重新开始前删除根目录 `STOP_REQUESTED`。如果系统崩溃留下 `.run.lock`，确认全部运行窗口已关闭后再删除它。不同方法不会共享被监督训练过的视觉权重。

## GPU 从原图重新提取

这是可选的**完整输入复现**，不是获得额外准确率的秘诀。它不更新预训练模型权重，理应与包内缓存结果接近。

1. 双击 `01_GPU_REEXTRACT.cmd`。首次需要安装固定版本 CUDA PyTorch 和模型库，并下载固定版本的预训练权重；默认先尝试国内镜像，失败时使用官方地址。基础回归包不需要这些下载。
2. 约需另备 15 GB 磁盘空间。首次模型下载约 6.3 GB；模型不放入普通 Git 仓库。它会校验固定 SHA，断点保留已完成批次，显存不足自动减少提取批大小。
3. 输出写入 `generated_features/`，不覆盖随包参考缓存。检查 `GPU_DIFFERENCE_REPORT.json` 与提取状态；完整、非 probe、来源和误差校验通过后，再双击 `03_GPU_COMPARE.cmd`。
4. 提取阶段可回传 `GPU_REEXTRACT_RESULTS.zip` 查看下载/编码日志；重新回归后，`TEAM_RESULTS.zip` 的状态会注明使用了重新提取的特征。

已有官方模型可在 `paths.json` 中指定，格式见 `paths.example.json`。不要直接编辑原特征、划分或价格来“通过检查”。

本机已验证真实数据回归流程及小样本输入编码；Windows 启动脚本完成静态与完整性检查；本机没有 PowerShell，语法解析测试跳过。交付前未在组员的 Windows/CUDA 硬件上实际执行，不能将这些检查写成 RTX 实测。

## 结果与提交

- `SCORES.csv`：完整方法的原始价格尺度 MSE、MAE、偏差。
- `SCORES_BY_FOLD.csv`、`OOF_PREDICTIONS.csv`：同折、逐图证据。
- `PARITY.json`：与已验证缓存结果的数值差异。
- `PAIRED_INTERVALS.json`：固定预测下的描述性分组重采样区间，不包含重新训练和选择模型带来的不确定性。
- `models/`：本轮拟合的数值回归参数，不含视觉模型权重；不使用 Python pickle。
- `submissions/SUBMISSION_baseline.csv` 到 `SUBMISSION_v3.csv`：各自全量 8,000 标签重拟合后的 3,000 行测试预测，列为 `imageid,price`。请自行选择提交，程序不会替你提交 Kaggle。

OOF 分数对应每折的小训练集模型，不是对使用全部 8,000 标签后的模型的独立评估。`inner_calibration`、`historical_development` 在本轮 OOF 对比中没有进入训练；最终生成提交时才把全部训练标签纳入。程序不读取 sample solution 中的价格。

## 如何通过 GitHub 交给组员

1. 建议使用**组内私有仓库**，邀请组员访问。将 `HOUSE_PRICE_TEAM_GITHUB_SOURCE.zip` 解压后的文件内容提交到仓库，`.gitignore` 已排除原图、特征、环境、权重和运行结果。
2. 在该仓库的 **Releases → Draft a new release** 中上传 `HOUSE_PRICE_TEAM_FULL.zip`，发布后把 Release 页面链接发给组员。
3. 让组员下载你上传的完整 ZIP；GitHub 自动生成的 `Source code (zip)` 只有代码，不包含运行资产。

普通 Git 文件不能超过 100 MiB；网页直接上传代码文件上限 25 MiB。完整包应作为 Release 附件，不要把完整 ZIP 拖进仓库文件列表。Release 每个附件需小于 2 GiB。依据：[大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Release 说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。

本完整包用于组内复现，含课程图片。**它不是学校最终提交 ZIP**：课程要求最终代码包不附训练/测试图片。整理正式提交时另按作业要求导出代码、README、测试预测及允许的复现材料。

## 课程与报告边界

作业允许现有模型/预训练特征；没有要求所有参数从零训练。SigLIP2 原模型采用图文预训练，本方案仅用视觉分支；课程禁用预训练 MLLM，但尚无老师对 SigLIP2 的明确许可结论，因此不要在报告写成“老师已批准”。纯视觉 V2 是保留的替代方案。

`docs/REPORT_PLAN.md` 给出研究结构；`AI_HANDOFF.json` 提供给组员 AI 读取的入口；`evidence/historical_METHODS.csv` 保留已经做过的冻结头、MLP、末四层微调与 LoRA 结果。未返回结果的全参数微调不能写进已完成成绩。
