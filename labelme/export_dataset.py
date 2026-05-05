import os
import os.path as osp
import json
import datetime
import uuid
import collections
from typing import Iterable, List, NamedTuple, Tuple

import imgviz
import numpy as np
from loguru import logger

from labelme.label_file import LabelFile
from labelme.utils import img_data_to_arr
from labelme.utils import shape as shape_utils

try:
    import pycocotools.mask  # type: ignore[import]
except Exception:
    pycocotools = None  # type: ignore[assignment]


class ExportResult(NamedTuple):
    """导出操作的统计结果."""
    success: int
    failed: List[str]   # 处理失败的图片路径列表
    no_label: List[str] # 没有找到标注文件的图片路径列表


def _get_label_json_path(image_path: str) -> str:
    """根据图片路径推断对应的 LabelMe JSON 文件路径.

    标注 JSON 保存在 <image_dir>/Label/<basename>.json。
    """
    label_dir = osp.join(osp.dirname(osp.abspath(image_path)), "Label")
    basename = osp.splitext(osp.basename(image_path))[0]
    return osp.join(label_dir, basename + ".json")


def _split_images_by_label(image_paths: Iterable[str]):
    """将图片列表拆分为「有标注」和「无标注」两组.

    返回:
        available  : list of (image_path, json_path) — 找到了对应 JSON 的图片
        no_label   : list of image_path — 没有找到 JSON 的图片
    """
    available = []
    no_label = []
    for img_path in image_paths:
        json_path = _get_label_json_path(img_path)
        if osp.exists(json_path):
            available.append((img_path, json_path))
        else:
            no_label.append(img_path)
            logger.warning("标注文件不存在，跳过: {}", json_path)
    return available, no_label


def _build_class_mapping_from_label_files(label_files: Iterable[str]) -> Tuple[dict, list]:
    """从一批 LabelMe JSON 中自动收集所有类别名，并构建 name->id 映射."""
    class_names_set = set()
    for lf in label_files:
        try:
            data = json.load(open(lf, "r", encoding="utf-8"))
        except Exception as e:
            logger.warning("读取标签文件失败，已跳过: {} ({})", lf, e)
            continue
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label:
                class_names_set.add(label)

    # 固定前两个保留类，与官方脚本保持兼容
    class_names = ["__ignore__", "_background_"] + sorted(class_names_set)
    class_name_to_id = {name: i - 1 for i, name in enumerate(class_names)}
    return class_name_to_id, class_names


def _load_image_array(label_file: LabelFile, image_path: str) -> np.ndarray:
    """从 LabelFile 加载图片数组，并确保返回 RGB（3通道）格式.

    JPEG 不支持透明通道（RGBA），导出前统一转为 RGB，避免保存报错。
    """
    import PIL.Image

    if label_file.imageData:
        arr = img_data_to_arr(label_file.imageData)
        if arr.ndim == 3 and arr.shape[2] == 4:
            # RGBA → RGB：将透明区域合成到白色背景
            pil_img = PIL.Image.fromarray(arr, mode="RGBA")
            background = PIL.Image.new("RGB", pil_img.size, (255, 255, 255))
            background.paste(pil_img, mask=pil_img.split()[3])
            return np.array(background)
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=2)
        return arr

    pil_img = PIL.Image.open(image_path)
    if pil_img.mode == "RGBA":
        background = PIL.Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        return np.array(background)
    return np.array(pil_img.convert("RGB"))


# ---------------------------------------------------------------------------
# COCO 导出
# ---------------------------------------------------------------------------

def export_to_coco(
    image_paths: Iterable[str],
    output_dir: str,
) -> ExportResult:
    """将给定图像对应的 LabelMe JSON 导出为 COCO 格式.

    Args:
        image_paths: 勾选的图片绝对路径列表。
        output_dir:  COCO 数据集根目录。

    Returns:
        ExportResult — 包含 success 数量、failed 列表、no_label 列表。
    """
    if pycocotools is None:
        raise RuntimeError(
            "未安装 pycocotools，无法导出 COCO 格式，请先运行: pip install pycocotools"
        )

    image_paths = list(image_paths)
    available, no_label = _split_images_by_label(image_paths)

    if not available:
        return ExportResult(success=0, failed=[], no_label=no_label)

    os.makedirs(output_dir, exist_ok=True)
    jpeg_dir = osp.join(output_dir, "JPEGImages")
    os.makedirs(jpeg_dir, exist_ok=True)

    now = datetime.datetime.now()
    coco_data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None)],
        images=[],
        type="instances",
        annotations=[],
        categories=[],
    )

    label_files = [jp for _, jp in available]
    class_name_to_id, _ = _build_class_mapping_from_label_files(label_files)
    for cls_name, cls_id in class_name_to_id.items():
        if cls_id < 0:
            continue
        coco_data["categories"].append(
            dict(supercategory=None, id=cls_id, name=cls_name)
        )

    out_ann_file = osp.join(output_dir, "annotations.json")

    failed: List[str] = []
    image_id = 0

    for img_path, lf_path in available:
        logger.info("COCO 导出: 处理 {}", lf_path)
        try:
            label_file = LabelFile(filename=lf_path)
            img = _load_image_array(label_file, img_path)

            base = osp.splitext(osp.basename(lf_path))[0]
            out_img_file = osp.join(jpeg_dir, base + ".jpg")
            imgviz.io.imsave(out_img_file, img)

            coco_data["images"].append(
                dict(
                    license=0,
                    url=None,
                    file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                )
            )

            masks = {}
            segmentations = collections.defaultdict(list)
            for shape in label_file.shapes:
                points = shape["points"]
                label = shape["label"]
                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")

                mask = shape_utils.shape_to_mask(img.shape[:2], points, shape_type)  # type: ignore[attr-defined]
                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)
                masks[instance] = masks.get(instance, mask) | mask

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                elif shape_type == "circle":
                    (x1, y1), (x2, y2) = points
                    r = np.linalg.norm([x2 - x1, y2 - y1])
                    n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                    i = np.arange(n_points_circle)
                    x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                    y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                    points = np.stack((x, y), axis=1).flatten().tolist()
                else:
                    points = np.asarray(points).flatten().tolist()

                segmentations[instance].append(points)

            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]
                if cls_id < 0:
                    continue

                mask = np.asfortranarray(mask.astype(np.uint8))
                rle = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(rle))
                bbox = pycocotools.mask.toBbox(rle).flatten().tolist()

                coco_data["annotations"].append(
                    dict(
                        id=len(coco_data["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )

            image_id += 1

        except Exception as e:
            logger.error("COCO 导出失败: {} — {}", img_path, e)
            failed.append(img_path)

    with open(out_ann_file, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False)

    result = ExportResult(success=image_id, failed=failed, no_label=no_label)
    logger.info(
        "COCO 导出完成 → 成功 {}，失败 {}，无标注 {}",
        result.success, len(result.failed), len(result.no_label),
    )
    return result


# ---------------------------------------------------------------------------
# VOC 导出
# ---------------------------------------------------------------------------

def export_to_voc(
    image_paths: Iterable[str],
    output_dir: str,
) -> ExportResult:
    """将给定图像对应的 LabelMe JSON 导出为 VOC 分割格式.

    Args:
        image_paths: 勾选的图片绝对路径列表。
        output_dir:  VOC 数据集根目录。

    Returns:
        ExportResult — 包含 success 数量、failed 列表、no_label 列表。
    """
    from labelme import utils as labelme_utils

    image_paths = list(image_paths)
    available, no_label = _split_images_by_label(image_paths)

    if not available:
        return ExportResult(success=0, failed=[], no_label=no_label)

    os.makedirs(output_dir, exist_ok=True)
    jpeg_dir = osp.join(output_dir, "JPEGImages")
    seg_class_dir = osp.join(output_dir, "SegmentationClass")
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(seg_class_dir, exist_ok=True)

    label_files = [jp for _, jp in available]
    class_name_to_id, _ = _build_class_mapping_from_label_files(label_files)
    class_names_clean = sorted(
        name
        for name, cid in class_name_to_id.items()
        if cid >= 0 and name not in ("__ignore__", "_background_")
    )

    out_class_names_file = osp.join(output_dir, "class_names.txt")
    with open(out_class_names_file, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names_clean))

    failed: List[str] = []
    success = 0

    for img_path, lf_path in available:
        logger.info("VOC 导出: 处理 {}", lf_path)
        try:
            label_file = LabelFile(filename=lf_path)
            img = _load_image_array(label_file, img_path)

            base = osp.splitext(osp.basename(lf_path))[0]
            out_img_file = osp.join(jpeg_dir, base + ".jpg")
            out_cls_file = osp.join(seg_class_dir, base + ".png")

            imgviz.io.imsave(out_img_file, img)

            cls, _ = shape_utils.shapes_to_label(  # type: ignore[attr-defined]
                img_shape=img.shape,
                shapes=label_file.shapes,
                label_name_to_value=class_name_to_id,
            )
            cls[cls < 0] = 0

            labelme_utils.lblsave(out_cls_file, cls)
            success += 1

        except Exception as e:
            logger.error("VOC 导出失败: {} — {}", img_path, e)
            failed.append(img_path)

    result = ExportResult(success=success, failed=failed, no_label=no_label)
    logger.info(
        "VOC 导出完成 → 成功 {}，失败 {}，无标注 {}",
        result.success, len(result.failed), len(result.no_label),
    )
    return result
