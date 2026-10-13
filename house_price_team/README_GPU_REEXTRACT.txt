GPU重新提取：用途、资源与复现边界
================================

日常复刻优先使用 assets/features 内已经验证的特征，直接运行相同回归实验。
GPU入口适合验证“从原始图片到特征”的完整流程；视觉模型保持冻结，所有参数
requires_grad=False，不使用房价标签，不进行视觉骨干微调，也不代表MSE会下降。
新特征位于 generated_features，原 assets/features 不会被覆盖。

入口
----
在已经装好依赖的环境运行：
  python team_extract.py --root .
或由 run_experiments.py 的 extract 模式调用 team_extract.main(['--root', root])。

默认必须检测到NVIDIA CUDA。没有默认总时限；可选 --max-seconds 1800。
在根目录创建 STOP_REQUESTED（也兼容STOP_REQUESTED.txt）可请求停止；已提交的批次保留在断点缓存。
继续前删除STOP文件，再运行同一入口。网络读操作最多可能等待30秒后响应。
CPU或MPS仅可显式运行少量机械校验，例如：
  python team_extract.py --device cpu --probe-count 2 --output probe_cpu
probe输出不是完整特征，不能当全量训练输入。

新机器会下载什么
--------------
名称                  固定模型                           文件尺寸(十进制)
register_large336     facebook/dinov2-with-registers-large  约1.218 GB
b448                  facebook/dinov2-base                  约0.346 GB
l336                  facebook/dinov2-large                 约1.218 GB
full                  google/siglip2-large-patch16-256      约3.526 GB

四个官方权重合计约6.31 GB；配置文件很小。依赖安装和图片/原特征另计。
建议为下载、断点和新旧特征至少留15 GB额外磁盘空间；安装CUDA PyTorch还需额外空间。
一次只加载一个视觉模型，FP32分批提取；显存不足会减小批量再试。
时间取决于显卡、网络与磁盘；不把本机小样本速度当成全量GPU运行保证。

权重不随此包重复传输。下载依次尝试 hf-mirror.com 和 huggingface.co，URL均固定
到MODEL_SOURCES.json中的完整commit，不追踪main；每个完整文件验证SHA256。
镜像只是传输渠道，SHA不匹配不会加载。支持HTTP Range断点下载。
SigLIP2官方完整文件含图像/文字两塔，但只读取vision_model.*，不会实例化文字塔。
若本地已有此前约1.264GB视觉子集权重，模块也支持其已锁定SHA256，减少下载。

复用本地模型
------------
在paths.json设置可选models字段，值是包含config.json和model.safetensors的文件夹：
{
  "models": {
    "register_large336": "D:/models/dinov2-with-registers-large",
    "b448": "D:/models/dinov2-base",
    "l336": "D:/models/dinov2-large",
    "full": "D:/models/siglip2-large-patch16-256"
  }
}
仅填写已有模型项；其他项仍自动下载。模型/config与固定SHA不符时明确失败，并保留
原文件。使用 --offline 禁止下载。也可 --paths 指向包外的路径配置，便于本地验证。
paths支持dataset（默认assets/dataset）、features（默认assets/features）、
rows（默认data/ROWS.csv）。路径相对于包根目录；绝对路径也可以。

严格沿用的输入与输出
------------------
DINO三组：EXIF校正，转RGB，等比BICUBIC缩放到448或336，中心补边；ImageNet
归一化后将补边区域置0。输出CLS与按有效像素面积加权的空间patch均值。
带register的Large模型必须跳过CLS后4个register token，不把它们当空间patch。
SigLIP2：EXIF校正，转RGB，全图BILINEAR拉伸到256×256，均值/标准差均为0.5。
输出attention pooler的1024维与所有256个patch平均的1024维，未经回归阶段L2归一化。
这不是最终版曾试过的75%中心裁剪；此包复刻研究中锁定的四层方法递进。

所有图片SHA256按canonical rows核对，特征NPZ固定字段为keys、image_sha256、features。
输出原始FP32特征；回归程序随后进行既定分块归一化和核岭训练。

如何判断成功
------------
EXTRACTION_STATUS.json 为 complete，表示所有所选分支完整完成；probe_complete
只表示小样本机械检查完成。GPU_DIFFERENCE_REPORT.json记录与原特征的最大绝对差、
RMSE、余弦相似度与FP32数值容差；过大差异会停下，避免把错误预处理送入回归。
成功后再用 generated_features 作为特征目录运行相同的四个回归配方，比较MSE。
跨硬件数值误差可能存在，不宣称GPU重提后能提高预测精度。

中断/失败
---------
保留generated_features/.cache以及models_cache中的partial文件，重跑可继续。
不要手工混合不同设备、模型或配置的缓存；缓存绑定源码、模型、图片和运行库。
已完整生成但来源不同的NPZ不会被覆盖，请显式指定新的 --output 目录。
EXTRACTION_STATUS.json和GPU_DIFFERENCE_REPORT.json会保留失败原因，批次缓存保留。

方法与课程边界
-------------
所有视觉模型来源与commit/SHA列在MODEL_SOURCES.json。SigLIP2视觉塔来自图文对比
预训练；技术复现不等于课程已明确许可。课程模型使用资格须与老师要求单独核对。
