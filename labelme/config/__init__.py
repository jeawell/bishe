import os.path as osp
import shutil

import yaml
from loguru import logger

here = osp.dirname(osp.abspath(__file__))


def update_dict(target_dict, new_dict, validate_item=None):
    for key, value in new_dict.items():
        if validate_item:
            validate_item(key, value)
        if key not in target_dict:
            logger.warning("Skipping unexpected key in config: {}".format(key))
            continue
        if isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict(target_dict[key], value, validate_item=validate_item)
        else:
            target_dict[key] = value


# -----------------------------------------------------------------------------


def get_default_config():
    # 构建默认配置文件路径，这里使用osp.join确保路径正确性
    config_file = osp.join(here, "default_config.yaml")

    # 打开并读取默认配置文件
    with open(config_file) as f:
        # 使用yaml安全加载方式解析配置文件内容
        config = yaml.safe_load(f)

    # 构建用户主目录下的配置文件路径
    user_config_file = osp.join(osp.expanduser("~"), ".labelmerc")

    # 如果用户配置文件不存在，则创建它
    if not osp.exists(user_config_file):
        try:
            # 将默认配置文件复制到用户目录
            shutil.copy(config_file, user_config_file)
        except Exception:
            # 如果复制失败，记录警告日志但不影响程序运行
            logger.warning("Failed to save config: {}".format(user_config_file))

    # 返回从默认配置文件中读取的配置字典
    return config

def validate_config_item(key, value):
    if key == "validate_label" and value not in [None, "exact"]:
        raise ValueError(
            "Unexpected value for config key 'validate_label': {}".format(value)
        )
    if key == "shape_color" and value not in [None, "auto", "manual"]:
        raise ValueError(
            "Unexpected value for config key 'shape_color': {}".format(value)
        )
    if key == "labels" and value is not None and len(value) != len(set(value)):
        raise ValueError(
            "Duplicates are detected for config key 'labels': {}".format(value)
        )


def get_config(config_file_or_yaml=None, config_from_args=None):
    # 1. 获取默认配置作为基础配置
    config = get_default_config()

    # 2. 如果提供了配置文件路径或YAML字符串，则加载并更新配置
    if config_file_or_yaml is not None:
        # 尝试直接将输入作为YAML内容加载
        config_from_yaml = yaml.safe_load(config_file_or_yaml)
        # 如果加载结果不是字典类型，说明输入可能是文件路径
        if not isinstance(config_from_yaml, dict):
            # 以文件形式打开并读取配置
            with open(config_file_or_yaml) as f:
                # 记录日志，显示正在从哪个文件加载配置
                logger.info("Loading config file from: {}".format(config_file_or_yaml))
                # 从YAML文件加载配置
                config_from_yaml = yaml.safe_load(f)
        # 使用从YAML获取的配置更新基础配置，并进行验证
        update_dict(config, config_from_yaml, validate_item=validate_config_item)

    # 3. 如果提供了命令行参数配置，则更新配置
    if config_from_args is not None:
        # 使用命令行参数更新配置，并进行验证
        update_dict(config, config_from_args, validate_item=validate_config_item)

    # 返回最终合并后的配置字典
    return config
