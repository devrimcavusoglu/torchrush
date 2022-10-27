import numpy as np
import PIL.Image
import torch


def pillow_to_numpy(img: PIL.Image.Image) -> np.ndarray:
    return np.array(img)


def pillow_to_torch(img: PIL.Image.Image) -> torch.Tensor:
    np_img = pillow_to_numpy(img)
    np_img = np_img.astype(np.float32)
    tensor = torch.from_numpy(np_img)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def numpy_to_pillow(img: np.ndarray, mode=None) -> PIL.Image.Image:
    return PIL.Image.fromarray(img, mode)
