import argparse
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0


def dice_score(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    return 2 * inter / (mask1.sum() + mask2.sum() + 1e-6)


def hungarian_match(gt_anns, dt_anns, iou_func):
    if len(gt_anns) == 0 or len(dt_anns) == 0:
        return []

    cost_matrix = np.zeros((len(gt_anns), len(dt_anns)))

    for i, gt in enumerate(gt_anns):
        for j, dt in enumerate(dt_anns):
            iou = iou_func(gt, dt)
            cost_matrix[i, j] = 1 - iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))


def evaluate(gt_path, dt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)
    coco_gt = COCO(None)
    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()
    with open(dt_path, 'r', encoding='utf-8') as f:
        dt_dataset = json.load(f)
    coco_dt = COCO(None)
    coco_dt.dataset = dt_dataset
    coco_dt.createIndex()

    results = {
        "bbox_iou": [],
        "mask_iou": [],
        "dice": [],
        "per_class": {}
    }

    for cat_id in coco_gt.getCatIds():
        results["per_class"][cat_id] = {
            "bbox_iou": [],
            "mask_iou": [],
            "dice": []
        }

    for img_id in coco_gt.imgs:
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])

        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        dt_anns = coco_dt.loadAnns(dt_ann_ids)

        gt_by_cat = {}
        dt_by_cat = {}

        for ann in gt_anns:
            gt_by_cat.setdefault(ann["category_id"], []).append(ann)

        for ann in dt_anns:
            dt_by_cat.setdefault(ann["category_id"], []).append(ann)

        for cat_id in gt_by_cat:
            if cat_id not in dt_by_cat:
                continue

            gt_list = gt_by_cat[cat_id]
            dt_list = dt_by_cat[cat_id]

            def bbox_iou_func(gt, dt):
                return bbox_iou(gt["bbox"], dt["bbox"])

            matches = hungarian_match(gt_list, dt_list, bbox_iou_func)

            for gi, di in matches:
                gt = gt_list[gi]
                dt = dt_list[di]

                iou_bbox = bbox_iou(gt["bbox"], dt["bbox"])
                results["bbox_iou"].append(iou_bbox)
                results["per_class"][cat_id]["bbox_iou"].append(iou_bbox)

                if "segmentation" in gt and "segmentation" in dt:
                    mask_gt = coco_gt.annToMask(gt)
                    mask_dt = coco_dt.annToMask(dt)

                    iou_mask = mask_iou(mask_gt, mask_dt)
                    dice = dice_score(mask_gt, mask_dt)

                    results["mask_iou"].append(iou_mask)
                    results["dice"].append(dice)

                    results["per_class"][cat_id]["mask_iou"].append(iou_mask)
                    results["per_class"][cat_id]["dice"].append(dice)

    print("\n=== Overall Results ===")
    print(f"Mean BBox IoU: {np.mean(results['bbox_iou']):.4f}")
    print(f"Mean Mask IoU: {np.mean(results['mask_iou']) if results['mask_iou'] else 0:.4f}")
    print(f"Mean Dice: {np.mean(results['dice']) if results['dice'] else 0:.4f}")

    print("\n=== Per-Class Results ===")
    for cat_id, stats in results["per_class"].items():
        cat_name = coco_gt.loadCats([cat_id])[0]["name"]
        print(f"\nClass: {cat_name}")
        print(f"  BBox IoU: {np.mean(stats['bbox_iou']) if stats['bbox_iou'] else 0:.4f}")
        print(f"  Mask IoU: {np.mean(stats['mask_iou']) if stats['mask_iou'] else 0:.4f}")
        print(f"  Dice: {np.mean(stats['dice']) if stats['dice'] else 0:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO 标注评估（支持 Hungarian、Dice、多类别）")
    parser.add_argument("--gt", type=str, required=True, help="人工标注 COCO JSON")
    parser.add_argument("--dt", type=str, required=True, help="半自动标注 COCO JSON")
    args = parser.parse_args()

    evaluate(args.gt, args.dt)
