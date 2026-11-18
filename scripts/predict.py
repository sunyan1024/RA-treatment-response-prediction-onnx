"""疗效预测（label-free 单例入口）。

输入：单个病人的 JSON（影像 + 掩码 + 23 项临床指标，均为原始值）。
输出：P(治疗有效)（5 折平均）+ 中文报告。

用法：
    python predict.py --patient patient.json [--threshold 0.5] [--out report.md] [--device cpu|cuda]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

from contract import joint_index, MAX_INSTANCES, MODEL_ID
from extract_radiomics import make_extractor, extract_features, align_to_schema

HERE = Path(__file__).resolve().parent
CONFIG = HERE
_ORT_THREADS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 与训练一致的影像预处理：RGB + Resize(299) + ToTensor + ImageNet 归一化
# 抗体定性结果的映射（与 clinical/preprocess.ipynb cell 5 一致）
PNO_MAP = {"P": 1.0, "N": 0.0, "O": -1.0, "1": 1.0, "0": 0.0, "-1": -1.0}




def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def coerce_clinical_value(v):
    """严格解析 P/N/O 或数值，不猜测单位、比较符、缺失值。"""
    if isinstance(v, bool) or v is None:
        raise ValueError("临床值不能是布尔值或 null")
    if isinstance(v, str):
        v = PNO_MAP.get(v.strip().upper(), v.strip())
    try:
        value = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"无法解析临床值：{v!r}；请使用不带单位或比较符的数值")
    if not np.isfinite(value):
        raise ValueError("临床值必须是有限数值")
    return value


def validate_scalers(schema, scalers):
    radio = scalers["radiomics"]
    if radio["columns"] != schema["radiomics_columns"]:
        raise ValueError("radiomics scaler 列顺序与 schema 不一致")
    size = len(schema["radiomics_columns"])
    for key in ("mean", "std"):
        a = np.asarray(radio[key], dtype=np.float64)
        if a.shape != (size,) or not np.isfinite(a).all():
            raise ValueError(f"radiomics {key} 无效")
        if key == "std" and (a <= 0).any():
            raise ValueError("radiomics std 必须大于 0")
    c = scalers["clinical"]
    if set(c["continuous"]) & set(c["discrete"]):
        raise ValueError("临床连续/离散字段重复")
    if set(c["continuous"]) | set(c["discrete"]) != set(schema["clinical_columns"]):
        raise ValueError("临床 scaler 字段与 schema 不一致")
    for name, v in c["continuous"].items():
        if not np.isfinite([v["mean"], v["std"]]).all() or v["std"] <= 0:
            raise ValueError(f"临床 scaler 无效：{name}")


def normalize_clinical(clinical_dict, schema, scalers):
    """23 项原始临床值 -> 标准化后的 float32[23]（按 schema.clinical_columns 顺序）。"""
    cols = schema["clinical_columns"]
    missing = [c for c in cols if c not in clinical_dict]
    if missing:
        raise ValueError(f"临床字段缺失 {len(missing)} 项: {missing}")

    cont = scalers["clinical"]["continuous"]   # {列名: {mean, std}}
    disc = set(scalers["clinical"]["discrete"])

    vec = np.zeros(len(cols), dtype=np.float32)
    for i, c in enumerate(cols):
        raw = coerce_clinical_value(clinical_dict[c])
        if c in disc:
            allowed = {0, 1} if c == "性别" else {-1, 0, 1}
            if raw not in allowed:
                raise ValueError(f"离散字段 {c} 不在允许编码中：{allowed}")
            vec[i] = raw
        else:
            vec[i] = (raw - cont[c]["mean"]) / cont[c]["std"]
    if not np.isfinite(vec).all():
        raise ValueError("标准化临床特征包含非有限值")
    return vec


def crop_and_pad(img_path, mask_path):
    """复现训练时的 image_cropped_padding：裁 ROI 外接框 -> 灰度 -> 补成正方形(黑底居中，边长=ROI 长边)。"""
    img = Image.open(img_path).convert("L")
    mask = Image.open(mask_path).convert("L")
    if img.size != mask.size:
        raise ValueError("影像与掩码尺寸不一致")
    mask_np = np.array(mask)
    labels = np.unique(mask_np[mask_np != 0])
    if len(labels) != 1 or int(labels[0]) not in (1, 255):
        raise ValueError("掩码必须为 0/1 或 0/255 的单 ROI 二值图")
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        raise ValueError(f"掩码无前景：{mask_path}")
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("ROI 外接框至少需要两行、两列像素")
    cropped = img.crop((x_min, y_min, x_max, y_max))
    w, h = cropped.size
    side = max(w, h)
    square = Image.new("L", (side, side), 0)
    square.paste(cropped, ((side - w) // 2, (side - h) // 2))
    return square


def preprocess_image(img_path, mask_path):
    img = crop_and_pad(img_path, mask_path).convert("RGB")
    return ((np.asarray(img.resize((299, 299), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2,0,1) / np.float32(255) - np.asarray(IMAGENET_MEAN,dtype=np.float32)[:,None,None]) / np.asarray(IMAGENET_STD,dtype=np.float32)[:,None,None])  # [3, 299, 299]


def predict_patient(patient, schema, scalers, device="cuda", threshold=0.5,
                    weights_dir=None, params_path=None):
    """核心流程：校验 -> radiomics + 影像 + 临床 -> 5 折集成 -> (prob, detail)。"""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold 必须在 [0, 1] 内")
    images = patient.get("images", [])
    if not 1 <= len(images) <= MAX_INSTANCES:
        raise ValueError(f"每位患者需要 1..{MAX_INSTANCES} 张影像")
    joint_indices = np.asarray([[joint_index(item.get("joint")) for item in images]], dtype=np.int64)
    validate_scalers(schema, scalers)
    weights_dir = Path(weights_dir) if weights_dir else HERE.parent / "models"
    params_path = params_path or CONFIG / "Params.yaml"
    for fold in range(1, 6):
        ckpt = weights_dir / f"fold_{fold}.onnx"
        if not ckpt.is_file():
            raise FileNotFoundError(f"缺少带关节嵌入、位置编码的三模态权重：{ckpt}。请先训练对应模型，不能使用旧版权重。")
    clinical_vec = normalize_clinical(patient["clinical"], schema, scalers)

    images = patient["images"]
    masks = patient.get("masks") or []
    if not images:
        raise ValueError("images 为空，至少需要 1 张影像")
    if len(masks) != len(images):
        raise ValueError(f"masks({len(masks)}) 与 images({len(images)}) 数量不一致")

    extractor = make_extractor(params_path)

    img_tensors, radio_vecs = [], []
    warnings = []
    for i, img_item in enumerate(images):
        img_path = img_item["path"]
        mask_path = masks[i]["path"]
        for path in (img_path, mask_path):
            with Image.open(path) as im:
                if im.mode != "L":
                    raise ValueError(f"仅支持 8 位单通道灰度图：{path}")
        img_tensors.append(preprocess_image(img_path, mask_path))  # 裁 ROI+补正方形

        feats = extract_features(extractor, img_path, mask_path)  # 原始图+掩码，自动识别 ROI 标签
        raw = align_to_schema(feats, schema["radiomics_columns"])  # [91] 原始值
        rmean = np.array(scalers["radiomics"]["mean"], dtype=np.float32)
        rstd = np.array(scalers["radiomics"]["std"], dtype=np.float32)
        normalized = (raw - rmean) / rstd
        if not np.isfinite(normalized).all():
            raise ValueError("标准化 radiomics 包含非有限值")
        radio_vecs.append(normalized)

    N = len(img_tensors)
    feeds = dict(images=np.stack(img_tensors)[None].astype(np.float32), radios=np.stack(radio_vecs)[None].astype(np.float32), clinical=clinical_vec[None], mask=np.zeros((1,N),dtype=bool), joint_indices=joint_indices)
    providers = [('CUDAExecutionProvider', {'use_tf32': '0'}), 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    if device == 'cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CUDAExecutionProvider unavailable')
    fold_probs=[]
    for fold in range(1,6):
        options=ort.SessionOptions()
        options.intra_op_num_threads=_ORT_THREADS
        session=ort.InferenceSession(str(weights_dir / f'fold_{fold}.onnx'),sess_options=options,providers=providers)
        session.disable_fallback()
        if device == 'cuda' and 'CUDAExecutionProvider' not in session.get_providers():
            raise RuntimeError('CUDA initialization failed')
        logits=session.run(None,feeds)[0]
        shifted=logits-logits.max(axis=1,keepdims=True)
        exp=np.exp(shifted)
        value=float((exp/exp.sum(axis=1,keepdims=True))[0,1])
        if not np.isfinite(value): raise ValueError('Non-finite probability')
        fold_probs.append(value)
        del session

    prob = float(np.mean(fold_probs))
    detail = {
        "engine": "onnxruntime", "package_version": "0.1.0", "device": device, "model_id": MODEL_ID,
        "joint_indices": joint_indices[0].tolist(),
        "patient_id": patient.get("patient_id"),
        "n_images": N,
        "joints": [img.get("joint", "") for img in images],
        "prob_effective": prob,
        "fold_probs": {f"fold_{i+1}": p for i, p in enumerate(fold_probs)},
        "threshold": threshold,
        "prediction": "有效" if prob >= threshold else "无效",
        "low_confidence": abs(prob - threshold) < 0.1,
        "warnings": warnings,
    }
    return prob, detail


def render_report(detail):
    L = []
    L.append("# 疗效预测报告")
    L.append("")
    L.append(f"- 病人编号：{detail['patient_id']}")
    L.append(f"- 影像数：{detail['n_images']} 张（关节：{'、'.join(detail['joints']) or '未标注'}）")
    L.append(f"- 治疗有效概率：**{detail['prob_effective']:.4f}**")
    L.append(f"- 判定：**{detail['prediction']}**（阈值 {detail['threshold']}）")
    if detail["low_confidence"]:
        L.append("- ⚠️ 概率接近阈值（±0.1）；这是阈值距离提示，不是校准后的置信度。")
    L.append("")
    L.append("各折概率：" + "、".join(f"{k}={v:.4f}" for k, v in detail["fold_probs"].items()))
    if detail["warnings"]:
        L.append("")
        L.append("## 输入告警")
        for w in detail["warnings"]:
            L.append(f"- {w}")
    return "\n".join(L)


def main():
    global _ORT_THREADS
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True, help="单个病人 JSON 路径")
    ap.add_argument("--schema", default=str(CONFIG / "schema.yaml"))
    ap.add_argument("--scalers", default=str(CONFIG / "scalers.json"))
    ap.add_argument("--params", default=str(CONFIG / "Params.yaml"))
    ap.add_argument("--weights-dir", default=None, help="默认本 skill 的 models 目录")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=int, default=2, help="CPU 线程数，默认 2")
    ap.add_argument("--json-out", help="结构化结果输出路径")
    ap.add_argument("--out", default=None, help="报告输出路径（不填则打印到 stdout）")
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("--threads 必须至少为 1")
    _ORT_THREADS = args.threads
    if args.device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        ap.error("CUDA 不可用，请使用 --device cpu")

    schema = load_yaml(args.schema)
    with open(args.scalers, encoding="utf-8") as f:
        scalers = json.load(f)

    threshold = args.threshold if args.threshold is not None else float(schema.get("threshold", 0.5))
    weights_dir = args.weights_dir or str(HERE.parent / "models")

    with open(args.patient, encoding="utf-8") as f:
        patient = json.load(f)

    patient_dir = Path(args.patient).resolve().parent
    for item in patient.get("images", []) + patient.get("masks", []):
        path = Path(item["path"]).expanduser()
        item["path"] = str(path if path.is_absolute() else patient_dir / path)

    _, detail = predict_patient(
        patient, schema, scalers, device=args.device, threshold=threshold,
        weights_dir=weights_dir, params_path=args.params,
    )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = render_report(detail)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"报告已写入 {args.out}")
    print(report)


if __name__ == "__main__":
    main()
