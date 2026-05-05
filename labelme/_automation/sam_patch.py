"""
运行时猴子补丁：为 osam 的三个 SAM 系列解码器注入可调 logit 阈值。

原理：osam 内部三处解码器都把 `masks > 0.0` 硬编码，
这里用同签名的替代函数覆盖那三个模块级别的函数引用，
使 `_logit_threshold` 生效。线程安全：Python float 读写由 GIL 保证原子性。
"""
from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime
from osam import types

# ------------------------------------------------------------------ #
#  全局参数                                                            #
# ------------------------------------------------------------------ #
_logit_threshold: float = 0.0


def set_logit_threshold(value: float) -> None:
    global _logit_threshold
    _logit_threshold = float(value)


def get_logit_threshold() -> float:
    return _logit_threshold


# ------------------------------------------------------------------ #
#  Patched SAM (sam:100m / sam:300m / sam:latest)                     #
# ------------------------------------------------------------------ #
def _sam_generate_mask(
    decoder_session: onnxruntime.InferenceSession,
    image_embedding: types.ImageEmbedding,
    prompt: types.Prompt,
    input_size: int,
) -> npt.NDArray[np.bool_]:
    from osam._models.sam._images import compute_scale_to_resize_image

    if prompt.points is None or prompt.point_labels is None:
        raise ValueError("Prompt must contain points and point_labels: %r" % prompt)

    onnx_coord: npt.NDArray[np.float32] = np.concatenate(
        [prompt.points, np.array([[0.0, 0.0]])], axis=0
    )[None, :, :]
    onnx_label: npt.NDArray[np.float32] = np.concatenate(
        [prompt.point_labels, np.array([-1])], axis=0
    )[None, :].astype(np.float32)

    _, new_height, new_width = compute_scale_to_resize_image(
        height=image_embedding.original_height,
        width=image_embedding.original_width,
        target_size=input_size,
    )
    onnx_coord = (
        onnx_coord.astype(float)
        * (
            new_width / image_embedding.original_width,
            new_height / image_embedding.original_height,
        )
    ).astype(np.float32)

    decoder_inputs = {
        "image_embeddings": image_embedding.embedding[None, :, :, :],
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.array([-1], dtype=np.float32),
        "orig_im_size": np.array(
            (image_embedding.original_height, image_embedding.original_width),
            dtype=np.float32,
        ),
    }

    masks, _, _ = decoder_session.run(None, decoder_inputs)
    mask: npt.NDArray[np.bool_] = masks[0, 0] > _logit_threshold
    return mask


# ------------------------------------------------------------------ #
#  Patched SAM2 (sam2:small / sam2:latest / sam2:large)               #
# ------------------------------------------------------------------ #
def _sam2_generate_mask(
    decoder_session: onnxruntime.InferenceSession,
    image_embedding: types.ImageEmbedding,
    prompt: types.Prompt,
    input_size: int,
) -> npt.NDArray[np.bool_]:
    input_point: npt.NDArray[np.float32] = np.array(prompt.points, dtype=np.float32)
    input_point = input_point / np.array(
        [
            image_embedding.original_width / input_size,
            image_embedding.original_height / input_size,
        ],
        dtype=np.float32,
    )
    input_label: npt.NDArray[np.float32] = np.array(
        prompt.point_labels, dtype=np.float32
    )

    decoder_inputs = {
        "image_embeddings": image_embedding.embedding[None],
        "high_res_features1": image_embedding.extra_features[0][None],
        "high_res_features2": image_embedding.extra_features[1][None],
        "point_coords": input_point[None],
        "point_labels": input_label[None],
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.array([0], dtype=np.float32),
        "orig_im_size": np.array(
            (image_embedding.original_height, image_embedding.original_width),
            dtype=np.int64,
        ),
    }

    masks, scores, _ = decoder_session.run(None, decoder_inputs)
    masks = cast(npt.NDArray[np.float32], masks)
    scores = cast(npt.NDArray[np.float32], scores)
    mask: npt.NDArray[np.bool_] = masks[0, np.argmax(scores)] > _logit_threshold
    return mask


# ------------------------------------------------------------------ #
#  Patched EfficientSAM (efficientsam:10m / efficientsam:latest)      #
# ------------------------------------------------------------------ #
def _efficientsam_generate_mask(
    decoder_session: onnxruntime.InferenceSession,
    image_embedding: types.ImageEmbedding,
    prompt: types.Prompt,
) -> npt.NDArray[np.bool_]:
    input_point: npt.NDArray[np.float32] = np.array(prompt.points, dtype=np.float32)
    input_label: npt.NDArray[np.float32] = np.array(
        prompt.point_labels, dtype=np.float32
    )

    decoder_inputs = {
        "image_embeddings": image_embedding.embedding[None, :, :, :],
        "batched_point_coords": input_point[None, None, :, :],
        "batched_point_labels": input_label[None, None, :],
        "orig_im_size": np.array(
            (image_embedding.original_height, image_embedding.original_width),
            dtype=np.int64,
        ),
    }

    masks, _, _ = decoder_session.run(None, decoder_inputs)
    mask: npt.NDArray[np.bool_] = masks[0, 0, 0, :, :] > _logit_threshold
    return mask


# ------------------------------------------------------------------ #
#  Apply                                                               #
# ------------------------------------------------------------------ #
def apply_patches() -> None:
    """将三个解码器的函数引用替换为带可调阈值的版本。"""
    import osam._models.sam._decoding as _sam_dec
    import osam._models.sam2._decoding as _sam2_dec
    import osam._models.efficientsam._decoding as _eff_dec

    _sam_dec.generate_mask_from_image_embedding = _sam_generate_mask
    _sam2_dec.generate_mask_from_image_embedding = _sam2_generate_mask
    _eff_dec.generate_mask_from_image_embedding = _efficientsam_generate_mask
