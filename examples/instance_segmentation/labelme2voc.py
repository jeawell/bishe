#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys
import math
import uuid
from typing import Optional

import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageDraw
from loguru import logger
import imgviz
import numpy as np

import labelme
from label_file import LabelFile


def shape_to_mask(
    img_shape: tuple[int, ...],
    points: list[list[float]],
    shape_type: Optional[str] = None,
    line_width: int = 10,
    point_size: int = 5,
) -> npt.NDArray[np.bool_]:
    mask = PIL.Image.fromarray(np.zeros(img_shape[:2], dtype=np.uint8))
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)  # type: ignore[arg-type]
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)  # type: ignore[arg-type]
    elif shape_type == "brush":
        draw.line(xy=xy, fill=1, width=line_width)  # type: ignore[arg-type]
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)  # type: ignore[arg-type]
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    elif shape_type in [None, "polygon"]:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)  # type: ignore[arg-type]
    else:
        raise ValueError(f"shape_type={shape_type!r} is not supported.")
    return np.array(mask, dtype=bool)



def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        line_width = shape["line_width"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask: npt.NDArray[np.bool_]
        if shape_type == "mask":
            if not isinstance(shape["mask"], np.ndarray):
                raise ValueError("shape['mask'] must be numpy.ndarray")
            mask = np.zeros(img_shape[:2], dtype=bool)
            (x1, y1), (x2, y2) = np.asarray(points).astype(int)
            mask[y1 : y2 + 1, x1 : x2 + 1] = shape["mask"]
        else:
            mask = shape_to_mask(img_shape[:2], points, shape_type,line_width)

        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir",default="data_annotated", help="Input annotated directory")
    parser.add_argument("--output_dir",default="data_annotated", help="Output dataset directory")
    parser.add_argument(
        "--labels",default="labels.txt", help="Labels file or comma separated text",
    )
    parser.add_argument(
        "--noobject", default=True,help="Flag not to generate object label", action="store_true"
    )
    parser.add_argument(
        "--nonpy",default=True,help="Flag not to generate .npy files", action="store_true"
    )
    parser.add_argument(
        "--noviz",default=True, help="Flag to disable visualization", action="store_true"
    )
    args = parser.parse_args()


    # 创建所有必要的输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"), exist_ok=True)
    if not args.nonpy:
        os.makedirs(osp.join(args.output_dir, "SegmentationClassNpy"), exist_ok=True)
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "SegmentationClassVisualization"), exist_ok=True)
    if not args.noobject:
        os.makedirs(osp.join(args.output_dir, "SegmentationObject"), exist_ok=True)
        if not args.nonpy:
            os.makedirs(osp.join(args.output_dir, "SegmentationObjectNpy"), exist_ok=True)
        if not args.noviz:
            os.makedirs(osp.join(args.output_dir, "SegmentationObjectVisualization"), exist_ok=True)
    print("Creating dataset:", args.output_dir)

    if osp.exists(args.labels):
        with open(args.labels, encoding='utf-8') as f:
            labels = [label.strip() for label in f if label]
    else:
        labels = [label.strip() for label in args.labels.split(",")]

    class_names = []
    class_name_to_id = {}
    for i, label in enumerate(labels):
        class_id = i - 1  # starts with -1
        class_name = label.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    try:
        for filename in sorted(glob.glob(osp.join(args.input_dir, "*.json"))):
            print("Generating dataset from:", filename)
            label_file = LabelFile(filename=filename)
            base = osp.splitext(osp.basename(filename))[0]

            # 定义所有输出文件路径
            out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
            out_clsp_file = osp.join(args.output_dir, "SegmentationClass", base + ".png")

            # 检查关键文件是否已存在
            skip_processing = osp.exists(out_img_file) and osp.exists(out_clsp_file)

            # 根据参数添加其他文件检查
            if skip_processing and not args.nonpy:
                out_cls_file = osp.join(args.output_dir, "SegmentationClassNpy", base + ".npy")
                skip_processing = skip_processing and osp.exists(out_cls_file)

            if skip_processing and not args.noviz:
                out_clsv_file = osp.join(args.output_dir, "SegmentationClassVisualization", base + ".jpg")
                skip_processing = skip_processing and osp.exists(out_clsv_file)

            if skip_processing and not args.noobject:
                out_insp_file = osp.join(args.output_dir, "SegmentationObject", base + ".png")
                skip_processing = skip_processing and osp.exists(out_insp_file)

                if skip_processing and not args.nonpy:
                    out_ins_file = osp.join(args.output_dir, "SegmentationObjectNpy", base + ".npy")
                    skip_processing = skip_processing and osp.exists(out_ins_file)

                if skip_processing and not args.noviz:
                    out_insv_file = osp.join(args.output_dir, "SegmentationObjectVisualization", base + ".jpg")
                    skip_processing = skip_processing and osp.exists(out_insv_file)

            # 如果所有文件都已存在，跳过处理
            if skip_processing:
                print(f"Skipping {filename} - output files already exist")
                continue

            img = labelme.utils.img_data_to_arr(label_file.imageData)
            imgviz.io.imsave(out_img_file, img)

            cls, ins = shapes_to_label(
                img_shape=img.shape,
                shapes=label_file.shapes,
                label_name_to_value=class_name_to_id,
            )
            ins[cls == -1] = 0  # ignore it.

            # class label
            labelme.utils.lblsave(out_clsp_file, cls)
            if not args.nonpy:
                np.save(osp.join(args.output_dir, "SegmentationClassNpy", base + ".npy"), cls)
            if not args.noviz:
                clsv = imgviz.label2rgb(
                    cls,
                    imgviz.rgb2gray(img),
                    label_names=class_names,
                    font_size=15,
                    loc="rb",
                )
                imgviz.io.imsave(osp.join(args.output_dir, "SegmentationClassVisualization", base + ".jpg"), clsv)

            if not args.noobject:
                # instance label
                out_insp_file = osp.join(args.output_dir, "SegmentationObject", base + ".png")
                labelme.utils.lblsave(out_insp_file, ins)
                if not args.nonpy:
                    np.save(osp.join(args.output_dir, "SegmentationObjectNpy", base + ".npy"), ins)
                if not args.noviz:
                    instance_ids = np.unique(ins)
                    instance_names = [str(i) for i in range(max(instance_ids) + 1)]
                    insv = imgviz.label2rgb(
                        ins,
                        imgviz.rgb2gray(img),
                        label_names=instance_names,
                        font_size=15,
                        loc="rb",
                    )
                    imgviz.io.imsave(osp.join(args.output_dir, "SegmentationObjectVisualization", base + ".jpg"), insv)
    except Exception as e:
        print(f"Error processing {filename if 'filename' in locals() else 'unknown file'}")
        print(e)
        # 可以选择继续处理下一个文件而不是退出
        # 如果需要完全停止，可以删除这行并保留pass
        # pass


if __name__ == "__main__":
    main()
