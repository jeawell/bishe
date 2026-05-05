import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np


def compute_mean_iou_bbox(coco_gt, coco_dt):
    """计算 bbox IoU 的平均值"""
    ious = []

    for img_id in coco_gt.imgs:
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])

        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        dt_anns = coco_dt.loadAnns(dt_ann_ids)

        # 简单匹配：按顺序匹配（适用于标注数量一致的情况）
        for gt, dt in zip(gt_anns, dt_anns):
            iou = compute_bbox_iou(gt["bbox"], dt["bbox"])
            ious.append(iou)

    return np.mean(ious) if len(ious) > 0 else 0


def compute_bbox_iou(boxA, boxB):
    """计算两个 COCO bbox 的 IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def compute_mean_iou_mask(coco_gt, coco_dt):
    """计算 segmentation mask IoU 的平均值"""
    ious = []

    for img_id in coco_gt.imgs:
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])

        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        dt_anns = coco_dt.loadAnns(dt_ann_ids)

        for gt, dt in zip(gt_anns, dt_anns):
            if "segmentation" not in gt or "segmentation" not in dt:
                continue

            mask_gt = coco_gt.annToMask(gt)
            mask_dt = coco_dt.annToMask(dt)

            inter = np.logical_and(mask_gt, mask_dt).sum()
            union = np.logical_or(mask_gt, mask_dt).sum()

            iou = inter / union if union > 0 else 0
            ious.append(iou)

    return np.mean(ious) if len(ious) > 0 else 0


def compute_coco_metrics(coco_gt, coco_dt):
    """使用 COCO 官方评估器计算 AP50、AP75 等指标"""
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 返回 AP50（指标 1）和 AP75（指标 2）
    return coco_eval.stats[1], coco_eval.stats[2]


def main(gt_path, dt_path):
    print("加载 COCO 标注文件...")
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dt_path)

    print("\n计算 bbox 平均 IoU...")
    mean_iou_bbox = compute_mean_iou_bbox(coco_gt, coco_dt)
    print(f"平均 bbox IoU: {mean_iou_bbox:.4f}")

    print("\n计算 mask 平均 IoU（如果有 segmentation）...")
    mean_iou_mask = compute_mean_iou_mask(coco_gt, coco_dt)
    print(f"平均 mask IoU: {mean_iou_mask:.4f}")

    print("\n计算 COCO 官方指标（AP50 / AP75）...")
    ap50, ap75 = compute_coco_metrics(coco_gt, coco_dt)
    print(f"AP50: {ap50:.4f}")
    print(f"AP75: {ap75:.4f}")

    print("\n=== 最终结果 ===")
    print(f"平均 bbox IoU: {mean_iou_bbox:.4f}")
    print(f"平均 mask IoU: {mean_iou_mask:.4f}")
    print(f"AP50: {ap50:.4f}")
    print(f"AP75: {ap75:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算两个 COCO 标注文件的 IoU 和 COCO 指标")
    parser.add_argument("--gt", type=str, required=True, help="人工标注 COCO JSON 路径")
    parser.add_argument("--dt", type=str, required=True, help="半自动标注 COCO JSON 路径")
    args = parser.parse_args()

    main(args.gt, args.dt)
