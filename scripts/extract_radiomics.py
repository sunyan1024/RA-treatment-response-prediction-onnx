"""pyradiomics 现场特征提取，并把结果对齐到冻结的 91 维 schema。

用法：
    python extract_radiomics.py --image <img.png> --mask <mask.png> \
        [--schema schema.yaml] [--params Params.yaml]

也可被 predict.py import 调用 `extract_features()` / `align_to_schema()`。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def load_schema(schema_path):
    with open(schema_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_extractor(params_path):
    """按训练时一致的设置初始化提取器：Params.yaml + enableAllImageTypes + enableAllFeatures。"""
    from radiomics import featureextractor
    extractor = featureextractor.RadiomicsFeatureExtractor(str(params_path))
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    return extractor


def extract_features(extractor, image_path, mask_path, label=None):
    """返回该影像的完整 pyradiomics 特征 dict（原始值，未标准化）。

    label 不指定时，自动取掩码中非零像素值作为 ROI 标签（训练集掩码为 0/1，
    外部集为 0/255，两者都能正确识别）。
    """
    import SimpleITK as sitk
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if label is None:
        roi = np.unique(sitk.GetArrayFromImage(mask))
        roi = roi[roi != 0]
        if roi.size == 0:
            raise ValueError(f"掩码全为 0，无 ROI：{mask_path}")
        label = int(roi.max())
    features = extractor.execute(image, mask, label=label)
    # execute 返回 OrderedDict：真正的特征值多为 shape(1,) 的 ndarray，
    # 诊断字段是 str/dict/tuple。只保留「单值数值」，统一转 float。
    out = {}
    for k, v in features.items():
        try:
            a = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if a.size == 1:
            out[k] = float(a.reshape(-1)[0])
    return out


def align_to_schema(feat_dict, radiomics_columns):
    """按冻结的 91 列顺序取特征；缺列或非有限值时报错，多余列忽略。返回 float32[91]。"""
    missing = [c for c in radiomics_columns if c not in feat_dict]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 个必需 radiomics 特征：{missing}")
    vec = np.array([feat_dict.get(c, 0.0) for c in radiomics_columns], dtype=np.float32)
    if not np.isfinite(vec).all():
        raise ValueError("radiomics 特征包含 NaN/Inf")
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--schema", default=str(Path(__file__).parent / "schema.yaml"))
    ap.add_argument("--params", default=str(Path(__file__).parent / "Params.yaml"))
    ap.add_argument("--label", type=int, default=None, help="ROI 标签；缺省自动取掩码非零值")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    extractor = make_extractor(args.params)
    feats = extract_features(extractor, args.image, args.mask, label=args.label)
    vec = align_to_schema(feats, schema["radiomics_columns"])

    if args.json:
        print(json.dumps({
            "image": args.image,
            "n_features": int(vec.shape[0]),
            "vector": vec.tolist(),
        }, ensure_ascii=False))
    else:
        print(f"提取到 {vec.shape[0]} 维 radiomics 特征（已对齐 schema）")
        print(" ".join(f"{v:.6g}" for v in vec))


if __name__ == "__main__":
    main()
