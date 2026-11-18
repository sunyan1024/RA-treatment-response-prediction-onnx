---
name: RA-predict-onnx
description: 使用本地五折ONNX Runtime模型，根据患者关节超声、ROI掩码、关节类型及临床指标执行疗效预测；用户要求使用该模型时使用。
---

# 关节炎疗效预测

以本SKILL.md所在的实际目录为SKILL_ROOT，不依赖agent当前工作目录。输入结构见[患者模板](references/patient.example.json)，其中的 `clinical` 对象列出了全部临床字段。

1. 确认本skill的环境与五折模型已准备好。环境缺失时使用本目录scripts/install.sh；权重从README中的网盘下载，文件名直接使用fold_1至fold_5，并用scripts/verify_weights.py校验SHA256。权重不随Git仓库提供，不使用随机模型或其他版本权重代替。
2. 收集全部23项原始临床值、8位单通道灰度PNG格式的关节超声图像、同尺寸ROI掩码（背景0、单个ROI值1或255），以及每张影像的关节类型。缺少值、单位或性别编码时询问，不猜测或用均值补齐。
3. 每位患者1–50张影像。关节支持膝/Knee、指/Knuckle、腕/Wrist、踝/Ankle、趾/Toe、肘/Elbow（中文使用“膝关节”等完整名称）。影像、掩码、关节按原训练文件名字典序同步排列；相对文件路径以patient.json所在目录为准。
4. 使用绝对路径执行：`bash <SKILL_ROOT>/scripts/run.sh --patient /absolute/patient.json --out /absolute/report.md --json-out /absolute/result.json`。程序自动完成ROI处理、组学提取及冻结标准化，不提前裁剪、归一化或重建scaler。
5. 仅在退出码为0时读取结果，报告五折平均概率、分类阈值和各折结果。错误不解释为治疗无效；阈值距离提示不是校准置信度，不据此生成治疗建议。

预测入口只读取本skill内的代码、配置和默认模型。安装脚本配置ONNX Runtime GPU；默认使用CUDA，不可用时明确报错，不自动退回CPU。
