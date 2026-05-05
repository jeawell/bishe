import numpy as np
from skimage import morphology, measure
from loguru import logger

# ------------------------------------------------------------------ #
#  全局参数（可由 UI 滑块动态调整）                                      #
# ------------------------------------------------------------------ #
_rdp_factor: float = 0.004


def set_rdp_factor(value: float) -> None:
    global _rdp_factor
    _rdp_factor = float(value)


def get_rdp_factor() -> float:
    return _rdp_factor


def compute_polygon_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    完全基于 skimage 的掩码后处理流程：
    1. 阈值化
    2. 形态学开闭运算
    3. 轮廓提取
    4. 面积筛选
    5. RDP 多边形拟合
    """

    # --- 1. 阈值化 ---
    # mask 可能是 float 或 bool，这里统一成 bool
    binary = mask > 0

    # --- 2. 形态学操作（开运算去噪 + 闭运算平滑） ---
    # selem = morphology.footprint_rectangle(3)
    selem = morphology.square(3)
    binary = morphology.opening(binary, selem)
    binary = morphology.closing(binary, selem)

    # --- 3. 轮廓提取 ---
    contours = measure.find_contours(binary, level=0.5)
    if len(contours) == 0:
        logger.warning("No contour found, returning empty polygon.")
        return np.empty((0, 2), dtype=np.float32)

    # --- 4. 面积筛选（取最大轮廓） ---
    def contour_area(cnt):
        # cnt 是 Nx2 的 y,x 坐标
        # 使用多边形面积公式
        x = cnt[:, 1]
        y = cnt[:, 0]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    contours = [c for c in contours if contour_area(c) > 10]
    if len(contours) == 0:
        logger.warning("Contours too small, returning empty polygon.")
        return np.empty((0, 2), dtype=np.float32)

    contour = max(contours, key=contour_area)

    # --- 5. RDP 多边形拟合 ---
    tolerance = _rdp_factor * max(np.ptp(contour, axis=0))  #越大拟合多边形点越少
    polygon = measure.approximate_polygon(contour, tolerance=tolerance)

    # 去掉重复点
    if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]

    # --- 6. 坐标格式转换（yx → xy） ---
    polygon = polygon[:, ::-1].astype(np.float32)

    # --- 7. 边界裁剪 ---
    h, w = mask.shape[:2]
    polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

    return polygon
