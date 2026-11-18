MODEL_ID = "inceptionv3_joint_position_multimodal_v1"
JOINT_TYPES = ("Knee", "Knuckle", "Wrist", "Ankle", "Toe", "Elbow")
JOINT_ALIASES = dict(zip(("膝关节", "指关节", "腕关节", "踝关节", "趾关节", "肘关节"), JOINT_TYPES))
MAX_INSTANCES = 50

def joint_index(value):
    if not isinstance(value, str):
        raise ValueError("每张影像必须提供 joint 关节类型")
    name = JOINT_ALIASES.get(value.strip(), value.strip())
    if name not in JOINT_TYPES:
        raise ValueError(f"不支持的关节类型 {value!r}，允许：{JOINT_TYPES} 或对应中文名称")
    return JOINT_TYPES.index(name)
