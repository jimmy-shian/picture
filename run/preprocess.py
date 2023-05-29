import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image


# def img_convert(input: np.ndarray) -> torch.Tensor:
#     """
#     input:
#     - type is np.ndarray.
#     - shape: (H, W, C), where C (Channel) is BGR (Blue, Green, Red) type.

#     output:
#     - type is torch.Tensor.
#     - shape: (C, H, W) where C (Channel) is RGB (Red, Green, Blue) type.
#     """
#     # Do your convert process.
#     return output

def img_convert(input: np.ndarray) -> torch.Tensor: #回傳的是torch.Tensor，用於測試
    # 将图像转换为灰度影像
    gray_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    # 调整图像大小为 (83, 84)
    resized_image = cv2.resize(gray_image, (83, 84))

    equalized_image = cv2.equalizeHist(resized_image)

    # 多次侵蚀后膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(equalized_image, kernel, iterations=2)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=2)

    # 调整图像大小为 (83, 84)
    resized_image = cv2.resize(dilated_image, (83, 84))

    # 将图像转换为张量并调整形状
    tensor_image = transforms.ToTensor()(resized_image)
    # tensor_image = torch.unsqueeze(tensor_image, 0)

    return tensor_image
