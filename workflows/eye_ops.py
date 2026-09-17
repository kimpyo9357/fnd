"""Eye operations retained from FND_eye/run.py in the original FND backup."""
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def color_slicing(img, thresholds=(145, 130, 170)):
    tb, tg, tr = thresholds
    b, g, r = cv2.split(img)

    def adjust_channel(channel, target):
        mean = np.mean(channel)
        offset = int(target - mean)
        return cv2.add(channel, offset)

    b_adj = adjust_channel(b, tb)
    g_adj = adjust_channel(g, tg)
    r_adj = adjust_channel(r, tr)

    return cv2.merge((b_adj, g_adj, r_adj))

def eye_crop(image, point1, point2,rotate):
    x1, y1 = point1
    point1_np = np.array([[x1], [y1], [1]])
    x2, y2 = point2
    point2_np = np.array([[x2], [y2], [1]])
    if x1 != x2:
        rotate = (y2-y1)/(x2-x1)
        rotate=math.degrees(math.atan(rotate))
        cx = int(x1+x2)//2
    else:
        cx = x1
        rotate = rotate * -90
    cy = int(y1 + y2) // 2
    mtrx = cv2.getRotationMatrix2D((cx,cy), rotate,1)
    point1_np = mtrx@point1_np
    point2_np = mtrx @point2_np
    x1 = point1_np[0][0]
    y1 = point1_np[1][0]
    x2 = point2_np[0][0]
    y2= point2_np[1][0]
    dst = cv2.warpAffine(image, mtrx,(0,0))
    dist = np.linalg.norm(np.array([x1,y1])-np.array([x2,y2]))/2 # distance - 눈 좌우 거리의 절반
    cx = (x1+x2)/2
    cy = (y1 + y2) / 2
    dst = dst[round(cy - dist):round(cy + dist), round(cx - dist):round(cx + dist), :] # destination, ouput img
    dst = cv2.resize(dst,(32,32))
    return dst

def load_model(weight_path="resnet18_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['0', '1']

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    print(weight_path)
    model.load_state_dict(torch.load(weight_path, map_location=device,weights_only=True))

    model = model.to(device).eval()
    return model, device

def open_predict(resnet, img_cv2):

    transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 학습에 맞게 조정 (원래 ResNet은 224x224)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])

    model, device = resnet

    if img_cv2 is None:
        raise ValueError("Image must be provided")

    # OpenCV: BGR → RGB
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # numpy → PIL.Image
    img_pil = Image.fromarray(img_rgb)

    # Transform 적용
    img_tensor = transform(img_pil)           # [3, 32, 32]
    img_tensor = img_tensor.unsqueeze(0)      # [1, 3, 32, 32]
    img_tensor = img_tensor.to(device)

    # 추론
    with torch.no_grad():
        output = model(img_tensor)

    pred_idx = output.argmax(dim=1).item()
    return pred_idx

def merge_close(arr, threshold=10):
    arr = arr.copy()  # 원본 보호

    between_zeros = np.where(
        (arr == 0) &
        (np.roll(arr, 1) == 1) &
        (np.roll(arr, -1) == 1)
    )[0]
    arr[between_zeros] = 0

    isolated_ones = np.where(
        (arr == 1) &
        (np.roll(arr, 1) == 0) &
        (np.roll(arr, -1) == 0)
    )[0]
    arr[isolated_ones] = 0

    one_index = np.where(arr == 1)[0]
    for idx in range(1, len(one_index)):
        if one_index[idx] - one_index[idx - 1] < threshold:
            arr[one_index[idx - 1]:one_index[idx] + 1] = 1

    return arr

def extract_midpoints_from_ones(arr: np.ndarray, max_len: int = 60) -> list:
    mask = arr == 1
    diff = np.diff(mask.astype(int))

    start_indices = np.where(diff == 1)[0] + 1
    end_indices = np.where(diff == -1)[0] + 1

    if mask[0]:
        start_indices = np.r_[0, start_indices]
    if mask[-1]:
        end_indices = np.r_[end_indices, len(arr)]

    midpoints = []

    for start, end in zip(start_indices, end_indices):
        length = end - start
        if length <= max_len:
            mid = start + length // 2
            midpoints.append(mid)
        else:
            for i in range(start, end, max_len):
                sub_start = i
                sub_end = min(i + max_len, end)
                mid = sub_start + (sub_end - sub_start) // 2
                midpoints.append(mid)
    
    return midpoints

def extract_around_midpoints(arr: np.ndarray, midpoints: list, window: int = 30) -> list:
    sub_arrays = []

    for mid in midpoints:
        start = mid - window
        end = mid + window
        if start < 0:
            shift = -start
            start = 0
            end = min(len(arr), end + shift)

        if end > len(arr):
            shift = end - len(arr)
            end = len(arr)
            start = max(0, start - shift)

        sub_arrays.append([start,end-1])

    return sub_arrays
