import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

def img_convert(input: np.ndarray) -> torch.Tensor: 
    try:
        gray_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    except:
        gray_image = cv2.resize(input, (83, 84))

    resized_image = cv2.resize(gray_image, (83, 84))
    equalized_image = cv2.equalizeHist(resized_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(equalized_image, kernel, iterations=2)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=2)

    resized_image = cv2.resize(dilated_image, (83, 84))

    tensor_image = transforms.ToTensor()(resized_image)

    return tensor_image