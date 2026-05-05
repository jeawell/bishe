from PyQt5 import QtCore
import osam
from labelme.shape import Shape
import imgviz
from labelme._automation import polygon_from_mask
from labelme._automation import sam_patch
import time

sam_patch.apply_patches()


class SamWorker(QtCore.QObject):
    """
    后台 SAM 推理工作者（无状态版本）。

    "只推最新"的逻辑由调用方（Canvas 主线程）负责：
    - 主线程维护 _sam_worker_busy 标志，只有 idle 时才 emit 信号；
    - 本类只做一件事：收到请求 → 推理 → 发回结果（包括 None）。
    - 必须发出 finished（哪怕结果为 None），以便主线程解锁 busy 标志。
    """

    finished = QtCore.pyqtSignal(object)   # Shape 或 None
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    @QtCore.pyqtSlot(object, str, str, object)
    def run_inference(self, shape, createMode, model_name, image_embedding):
        try:
            req = osam.types.GenerateRequest(
                model=model_name,
                image_embedding=image_embedding,
                prompt=osam.types.Prompt(
                    points=[[p.x(), p.y()] for p in shape.points],
                    point_labels=shape.point_labels,
                ),
            )
            start = time.perf_counter()
            response = osam.apis.generate(req)
            end = time.perf_counter()
            print(f"推理耗时: {end - start:.3f}s")
            if not response.annotations:
                self.finished.emit(None)
                return

            ann = response.annotations[0]

            if createMode == "ai_mask":
                if ann.bounding_box is None:
                    y1, x1, y2, x2 = imgviz.instances.mask_to_bbox(
                        [ann.mask]
                    )[0].astype(int)
                else:
                    y1 = ann.bounding_box.ymin
                    x1 = ann.bounding_box.xmin
                    y2 = ann.bounding_box.ymax
                    x2 = ann.bounding_box.xmax

                shape.setShapeRefined(
                    shape_type="mask",
                    points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                    point_labels=[1, 1],
                    mask=ann.mask[y1 : y2 + 1, x1 : x2 + 1],
                )

            elif createMode in ("ai_polygon", "ai_bbox"):
                pts = polygon_from_mask.compute_polygon_from_mask(ann.mask)
                if len(pts) >= 2:
                    shape.setShapeRefined(
                        shape_type="polygon",
                        points=[QtCore.QPointF(x, y) for x, y in pts],
                        point_labels=[1] * len(pts),
                    )

            self.finished.emit(shape)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(None)  # 必须发出以解锁主线程的 busy 标志
