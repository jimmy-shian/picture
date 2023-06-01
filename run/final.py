from pathlib import Path

# from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2
import time
# ==========================================================================================================
from preprocess import img_convert
device = 'cpu'

model_path = Path('./hiragana.pt')
# ==========================================================================================================

assert model_path.exists() is True

trained_model = torch.jit.load(model_path, map_location=device)  # in_shape: (N, C, H, W)
cate = {
    'A': 0, 'BA': 1, 'CHI': 2, 'DA': 3, 'E': 4, 'FU': 5, 'HA': 6, 'HE': 7, 'HI': 8, 'HO': 9, 'I': 10, 'JI': 11,
    'KA': 12, 'KE': 13, 'KI': 14, 'KO': 15, 'KU': 16, 'MA': 17, 'ME': 18, 'MI': 19, 'MO': 20, 'MU': 21, 'N': 22,
    'NA': 23, 'NE': 24, 'NI': 25, 'NO': 26, 'NU': 27, 'O': 28, 'PI': 29, 'RA': 30, 'RE': 31, 'RI': 32, 'RO': 33,
    'RU': 34, 'SA': 35, 'SE': 36, 'SHI': 37, 'SO': 38, 'SU': 39, 'TA': 40, 'TE': 41, 'TO': 42, 'TSU': 43, 'U':
    44, 'WA': 45, 'WO': 46, 'YA': 47, 'YO': 48, 'YU': 49
}
inv_cate = {v: k for k, v in cate.items()}

# =============================================
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# while(True):
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     frame = cv2.resize(frame, (960, 540))  # (W, H)
#     cv2.imshow('live', frame)
#     if cv2.waitKey(1) == ord('q'):
#         image = frame
#         break
# cap.release()
# cv2.destroyAllWindows()
# =========================================
# import cv2

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# # 设置绿框的参数
# box_width = 249
# box_height = 252
# box_color = (0, 255, 0)  # 绿色 (BGR 格式)
# box_thickness = 2

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break

#     frame = cv2.resize(frame, (960, 540))  # 调整窗口大小 (W, H)

#     # 计算绿框的位置
#     frame_height, frame_width, _ = frame.shape
#     box_x = int((frame_width - box_width) / 2)
#     box_y = int((frame_height - box_height) / 2)
#     box_x_end = box_x + box_width
#     box_y_end = box_y + box_height

#     # 绘制绿框
#     cv2.rectangle(frame, (box_x, box_y), (box_x_end, box_y_end), box_color, box_thickness)

#     cv2.imshow('live', frame)
#     if cv2.waitKey(1) == ord('q'):
#         cv2.destroyAllWindows()
#         image = frame[box_y:box_y_end, box_x:box_x_end]
#         cv2.imshow('live2', image)
#         cv2.waitKey(0)  # 等待任意按键继续执行
#         break

# cap.release()
# cv2.destroyAllWindows()
# =============================================================
import cv2

# # 讀取圖片
frame = cv2.imread('C:\\Users\\Administrator\\Desktop\\ahh\\testdata\\757.jpg')

# frame = cv2.resize(frame, (252, 249))
# cv2.imshow('live2', frame)
# cv2.waitKey(0)  # 等待任意按键继续执行

# ================================================================

trans_image = img_convert(frame)
# trans_image = img_convert(image)
# print(trans_image)


start_time = time.time()

pred = trained_model(trans_image.unsqueeze(0)) #
pred_label = pred.max(1)[1]

end_time = time.time()
prediction_time = end_time - start_time
print("預測時間:", prediction_time, "秒")

print(pred_label)
print([inv_cate[int(element)] for element in pred_label])
