"""Whisker operations retained from FND_whisker in the original FND backup."""
import math
from math import atan2
import cv2
import numpy as np
from mmdet.apis import inference_detector


def segment(img,model,y):
    global count, ann_id
    result = inference_detector(model, img)
    bbox_result, segm_result = result

    bbox_result = bbox_result[0]
    segm_result = segm_result[0]

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    out = []

    out_ori = result

    # Vectorized and filtered loop
    for i, (bbox, seg) in enumerate(zip(bbox_result, segm_result)):
        temp = {}
        temp['bbox'] = bbox

        x_min, y_min, x_max, y_max, score = bbox
        
        x_min = math.floor(x_min)
        x_max = math.ceil(x_max)
        y_min = math.floor(y_min)
        y_max = math.ceil(y_max)

        if score < 0.01:
            continue
        mask2 = seg.astype(np.uint8) * 255
        mask_crop = mask2[y_min:y_max, x_min:x_max]

        contours, hierarchy = cv2.findContours(mask_crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        

        if len(contours) > 1:
            area = [cv2.contourArea(cnt) for cnt in contours]
            contours = [contours[area.index(max(area))]]

            
        if len(contours) != 0:
            angle, cntr = get_orientation(contours[0], mask_crop)

            if angle == None:
                continue
            
            cntr = list(cntr)

            cntr[0] += x_min
            cntr[1] += y_min

            temp['PCA'] = (angle, cntr)

            y_target = get_y_from_x(mask.shape[1]//2, cntr[0], cntr[1], angle)
            if y[0]<y_target<y[1]:
                # cv2.line(img,cntr,(mask.shape[1]//2,int(y_target)),(0, 0, 255), 1)
                # cv2.imwrite(f'./mask/{i}.jpg',img)
                out.append(temp)

                mask = np.maximum(mask, mask2)  # Faster than cv2.bitwise_or

            # height, width, c = img.shape
            # p0, p1 = get_p0_p1(cntr[0], cntr[1], angle, height)
            # print(p0,p1)

            # cv2.line(img,p0,p1,(0, 255, 0), 1)


        
        # contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for i, contour in enumerate(contours, start=1):
        #     # 좌표 변환
        #     coords = contour[:, 0, :]  # (N,2)
            
        #     # 1️⃣ 최소 좌표 개수 (3개 이상이어야 polygon 가능)
        #     if len(coords) < 3:
        #         continue
            
        #     # polygon 생성

        #     polygon = [(float(x), float(y)) for x, y in coords]
            
        #     ann = polygon_to_annotation(bbox, polygon, ann_id=ann_id, image_id=img_id, category_id=1)
        #     out.append(ann)
        #     ann_id += 1
        # mask = np.maximum(mask, mask2)  # Faster than cv2.bitwise_or
        # out_bbox.append(bbox)
    return mask, out, out_ori

def angle_between_points(x1, y1, x2, y2):
    # 차이값
    dx = x2 - x1
    dy = y2 - y1
    
    # atan2는 사분면까지 고려해서 각도를 반환 (라디안 단위)
    angle_rad = math.atan2(dy, dx)
    
    # 도(degree)로 변환
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def transform_mask_and_points(mask, angle_deg, tx, ty, points=None, center=None):
    """
    mask      : 입력 마스크 (numpy array)
    angle_deg : 회전 각도 (deg)
    tx, ty    : 이동량
    points    : 변환할 좌표 리스트 [(x1, y1), (x2, y2), ...]
    center    : 회전 중심 (None이면 mask 중심)
    """
    h, w = mask.shape[:2]
    
    # 회전 중심 지정 (기본: 이미지 중심)
    if center is None:
        center = (w // 2, h // 2)

    # --- 이동 행렬 ---
    M_shift = np.float32([[1, 0, tx],
                          [0, 1, ty]])
    shifted_mask = cv2.warpAffine(mask, M_shift, (w, h))

    # --- 회전 행렬 (2x3) ---
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)

    # --- 최종 마스크 변환 ---
    transformed_mask = cv2.warpAffine(
        shifted_mask, M_rot, (w, h), 
        flags=cv2.INTER_NEAREST, borderValue=0
    )

    transformed_points = None
    if points is not None:
        # numpy 형태로 좌표 변환 준비
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        # 먼저 이동 적용
        pts_shifted = cv2.transform(pts, M_shift)
        # 그다음 회전 적용
        pts_transformed = cv2.transform(pts_shifted, M_rot)
        
        # numpy -> list로 변환
        transformed_points = pts_transformed.reshape(-1, 2).tolist()

    return transformed_mask, transformed_points


def get_orientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    # cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    try:
        p1 = (round(cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0]), round(cntr[1] + 0.02 *  eigenvectors[0,1] * eigenvalues[0,0]))
        p2 = (round(cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0]), round(cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0]))
    except:
        # print(eigenvectors, eigenvalues)
        # print("PCA part")
        # cv2.imshow('',img)
        # cv2.waitKey()
        return None, None

    # draw_axis(img, cntr, p1, (0, 0, 255), 1)
    angle2 = [p1[0]-cntr[0],p1[1]-cntr[1]]
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    angle2 = atan2(angle2[1],angle2[0])
    return angle, cntr

def get_y_from_x(x_target, x, y, w):
    
    y_target = w * (x_target - x) + y
    return y_target
