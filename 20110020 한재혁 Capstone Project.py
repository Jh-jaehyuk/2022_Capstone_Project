import math
import cv2
import numpy as np
import mediapipe as mp
import matplotlib
import matplotlib.pyplot as plt
import csv
import datetime as date
import time
import os

import seaborn as sns
import pandas as pd
import pyautogui
import requests
from io import BytesIO
from PIL import Image
import urllib.request
import PIL

import tkinter as tk
from tkinter import *
import tkinter.font as font
from tkinter import filedialog
from datetime import datetime
from PIL import ImageTk,Image
import urllib

from multiprocessing import Process

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles
#=======================================================================================================================
# 변수 초기화
# 오른쪽
right_lower_point = 1
right_upper_point = 1
right_wrist_point = 1
right_neck_point = 1
right_body_point = 1
right_leg_point = 1
right_twist_point = 1

right_lower_plus = 0
right_upper_plus = 0
right_wrist_plus = 0
right_neck_plus = 0
right_body_plus = 0
right_weight_plus = 0
right_muscle_plus = 0

right_lower_page = 1
right_upper_page = 1
right_wrist_page = 1
right_neck_page = 1
right_body_page = 1
right_leg_page = 1
right_weight_page = 1
right_muscle_page = 1

right_Score_A = 0
right_Score_B = 0
right_Score_Grand = 0

right_upper_count = 0
right_lower_count = 0
right_wrist_count = 0
right_body_count = 0
right_neck_count = 0

# 왼쪽
left_lower_point = 1
left_upper_point = 1
left_wrist_point = 1
left_neck_point = 1
left_body_point = 1
left_leg_point = 1
left_twist_point = 1

left_lower_plus = 0
left_upper_plus = 0
left_wrist_plus = 0
left_neck_plus = 0
left_body_plus = 0
left_weight_plus = 0
left_muscle_plus = 0

left_lower_page = 1
left_upper_page = 1
left_wrist_page = 1
left_neck_page = 1
left_body_page = 1
left_leg_page = 1
left_weight_page = 1
left_muscle_page = 1

left_Score_A = 0
left_Score_B = 0
left_Score_Grand = 0

left_upper_count = 0
left_lower_count = 0
left_wrist_count = 0
left_body_count = 0
left_neck_count = 0

#=======================================================================================================================

# 자동캡쳐 변수 초기화
img_counter_left_upper = 0
img_counter_left_lower = 0
img_counter_left_wrist = 0
img_counter_left_neck = 0
img_counter_left_body = 0
img_counter_left_grand_score = 0

img_counter_right_upper = 0
img_counter_right_lower = 0
img_counter_right_wrist = 0
img_counter_right_neck = 0
img_counter_right_body = 0
img_counter_right_grand_score = 0

# 캡쳐 시간 조절
right_upper_capture_time = 0
right_upper_capture_time2 = 0

right_lower_capture_time = 0
right_lower_capture_time2 = 0

right_neck_capture_time = 0
right_neck_capture_time2 = 0

right_wrist_capture_time = 0
right_wrist_capture_time2 = 0

right_body_capture_time = 0
right_body_capture_time2 = 0

right_grand_capture_time = 0
right_grand_capture_time2 = 0

left_upper_capture_time = 0
left_upper_capture_time2 = 0

left_lower_capture_time = 0
left_lower_capture_time2 = 0

left_neck_capture_time = 0
left_neck_capture_time2 = 0

left_wrist_capture_time = 0
left_wrist_capture_time2 = 0

left_body_capture_time = 0
left_body_capture_time2 = 0

left_grand_capture_time = 0
left_grand_capture_time2 = 0

## 평균값
right_upper_point_live = []
right_lower_point_live = []
right_wrist_point_live = []
right_neck_point_live = []
right_body_point_live = []
right_leg_point_live = []
right_grand_point_live = []

left_upper_point_live = []
left_lower_point_live = []
left_wrist_point_live = []
left_neck_point_live = []
left_body_point_live = []
left_leg_point_live = []
left_grand_point_live = []

'''
# Url image
url_R = "https://i.postimg.cc/g03B3z4W/bodypoint_R.jpg"
urllib.request.urlretrieve(url_R, "bodypoint_R.jpg")

url_L = "https://i.postimg.cc/YS8Y5TCF/bodypoint_L.jpg"
urllib.request.urlretrieve(url_L, "bodypoint_L.jpg")
'''

bodypoint_R_Img_url = 'https://i.postimg.cc/g03B3z4W/bodypoint-R.jpg'
bodypoint_L_Img_url = 'https://i.postimg.cc/YS8Y5TCF/bodypoint-L.jpg'

# Image open
R_image_nparray = np.asarray(bytearray(requests.get(bodypoint_R_Img_url).content), dtype=np.uint8)
L_image_nparray = np.asarray(bytearray(requests.get(bodypoint_L_Img_url).content), dtype=np.uint8)

#=======================================================================================================================
# 폴더 생성하기
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. '+ directory)

createFolder("C:/Users/{}/Desktop/My Program".format(os.getlogin()))

left_path = "C:/Users/{}/Desktop/My Program/left".format(os.getlogin())
folderlist = ['Record', 'Captured', 'Auto', 'Graph', 'Point']
folderlist2 = []

createFolder("C:/Users/{}/Desktop/My Program/Left".format(os.getlogin()))

for i in folderlist :
    folderlist2.append(left_path + '/{}'.format(i))

for k in folderlist2 :
    createFolder(k)


right_path = "C:/Users/{}/Desktop/My Program/Right".format(os.getlogin())
folderlist = ['Record', 'Captured', 'Auto', 'Graph', 'Point']
folderlist2 = []

createFolder("C:/Users/{}/Desktop/My Program/Right".format(os.getlogin()))

for i in folderlist :
    folderlist2.append(right_path + '/{}'.format(i))

for k in folderlist2 :
    createFolder(k)

#=======================================================================================================================
#
def detectPose(image, pose):
    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(imageRGB)

    height, width, _ = image.shape

    landmarks = []

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    return output_image, landmarks

# 각도 계산
def calculateAngle(landmark1, landmark2, landmark3):

    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # if angle < 0:
    #     angle += 360

    return angle

def calculateAngle_left(landmark1, landmark2, landmark3):

    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y2 - y1, x2 - x1))

    # if angle < 0:
    #     angle += 360

    return angle

def classifyPose_right(landmarks, output_image):
    global right_lower_point
    global right_upper_point
    global right_wrist_point
    global right_neck_point
    global right_body_point
    global right_leg_point

    global right_lower_plus
    global right_upper_plus
    global right_wrist_plus
    global right_neck_plus
    global right_body_plus
    global right_leg_plus

    global right_Score_A, right_Score_B, right_Score_Grand

    right_elbow_angle = 180 - calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])


    right_shoulder_angle = -1 * (calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]))


    right_body = 180 + calculateAngle((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1] + 200,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][2]),
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]) # 오른쪽 몸통 각도


    right_wrist = 180 - calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value])  # 오른쪽 손목 각도


    right_neck = 160 + calculateAngle((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1] + 200,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][2]),
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]) # 오른쪽 목 각도


    """RULA"""

    # Upper arm 상완
    right_upper_angle = round(right_shoulder_angle, 1)  # 오른쪽 상완, 소숫점 두번째 자리에서 반올림

    # Lower arm 전완
    right_lower_angle = round(right_elbow_angle, 1)  # 오른쪽 전완, 소숫점 두번째 자리에서 반올림

    # Wrist 손목
    right_wrist_angle = round(right_wrist, 1)  # 오른쪽 손목, 소숫점 두번째 자리에서 반올림

    # Neck 목
    right_neck_angle = round(right_neck, 1)  #

    # body 몸통
    right_body_angle = round(right_body, 1)  # 오른쪽 몸통, 소숫점 두번째 자리에서 반올림

    # Upper Arm Point 상완 점수
    # ----------------------------------------------------------------------------------------------------------------

    right_upper_x = int((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] + landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][0]) / 2)
    right_upper_y = int((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1] + landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][1]) / 2)

    if right_upper_angle > -10 and right_upper_angle < 20:
        right_upper_point = 1 + right_upper_plus
        cv2.circle(output_image, (right_upper_x, right_upper_y), 10, (0, 255, 0), -1)

    elif right_upper_angle > 20 and right_upper_angle < 45:
        right_upper_point = 2 + right_upper_plus
        cv2.circle(output_image, (right_upper_x, right_upper_y), 10, (0, 255, 255), -1)

    elif right_upper_angle > 45 and right_upper_angle < 90:
        right_upper_point = 3 + right_upper_plus
        cv2.circle(output_image, (right_upper_x, right_upper_y), 10, (0, 165, 255), -1)

    else:
        right_upper_point = 4 + right_upper_plus
        cv2.circle(output_image, (right_upper_x, right_upper_y), 10, (0, 0, 255), -1)

    global right_upper_point_live
    right_upper_point_live.append(right_upper_point)
    right_upper_point_avg = (sum(right_upper_point_live) / len(right_upper_point_live))
    right_upper_point_avg = round(right_upper_point_avg, 1)


    # Lower Arm Point 전완 점수
    # ----------------------------------------------------------------------------------------------------------------

    right_lower_x = int((landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][0] + landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0]) / 2)
    right_lower_y = int((landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][1] + landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]) / 2)

    if right_lower_angle > 60 and right_lower_angle < 100:
        right_lower_point = 1 + right_lower_plus
        cv2.circle(output_image, (right_lower_x, right_lower_y), 10, (0, 255, 0), -1)

    else:
        right_lower_point = 2 + right_lower_plus
        cv2.circle(output_image, (right_lower_x, right_lower_y), 10, (0, 0, 255), -1)

    global right_lower_point_live
    right_lower_point_live.append(right_lower_point)
    right_lower_point_avg = (sum(right_lower_point_live) / len(right_lower_point_live))
    right_lower_point_avg = round(right_lower_point_avg, 1)

    # Wrist Point 손목 점수
    # ----------------------------------------------------------------------------------------------------------------

    right_wrist_twist = 1

    if right_wrist_angle > -1 and right_wrist_angle < 1:
        right_wrist_point = 1 + right_wrist_plus
        cv2.circle(output_image, (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]), 10, (0, 255, 0), -1)

    elif (right_wrist_angle < -1 and right_wrist_angle > -15) or (right_wrist_angle > 1 and right_wrist_angle < 15):
        right_wrist_point = 2 + right_wrist_plus
        cv2.circle(output_image, (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]), 10, (0, 255, 255), -1)

    elif right_wrist_angle > 15 or right_wrist_angle < -15:
        right_wrist_point = 3 + right_wrist_plus
        cv2.circle(output_image, (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]), 10, (0, 0, 255), -1)

    global right_wrist_point_live
    right_wrist_point_live.append(right_wrist_point)
    right_wrist_point_avg = (sum(right_wrist_point_live) / len(right_wrist_point_live))
    right_wrist_point_avg = round(right_wrist_point_avg, 1)

    # Neck Point 목 점수
    # ----------------------------------------------------------------------------------------------------------------

    right_neck_x = int((landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value][0] + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / 2)
    right_neck_y = int((landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value][1] + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]) / 2)

    if right_neck_angle > -3 and right_neck_angle < 10:
        right_neck_point = 1 + right_neck_plus
        cv2.circle(output_image, (right_neck_x, right_neck_y), 10, (0, 255, 0), -1)

    elif right_neck_angle > 10 and right_neck_angle < 20:
        right_neck_point = 2 + right_neck_plus
        cv2.circle(output_image, (right_neck_x, right_neck_y), 10, (0, 255, 255), -1)

    elif right_neck_angle > 20:
        right_neck_point = 3 + right_neck_plus
        cv2.circle(output_image, (right_neck_x, right_neck_y), 10, (0, 165, 255), -1)

    elif right_neck_angle < -3:
        right_neck_point = 4 + right_neck_plus
        cv2.circle(output_image, (right_neck_x, right_neck_y), 10, (0, 0, 255), -1)

    global right_neck_point_live
    right_neck_point_live.append(right_neck_point)
    right_neck_point_avg = (sum(right_neck_point_live) / len(right_neck_point_live))
    right_neck_point_avg = round(right_neck_point_avg, 1)


        # Body Point 몸통 점수
    # ----------------------------------------------------------------------------------------------------------------

    right_body_x = int((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0] + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0]) / 2)
    right_body_y = int((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1] + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]) / 2)

    if right_body_angle >= -10 and right_body_angle < 0.5:
        right_body_point = 1 + right_body_plus
        cv2.circle(output_image, (right_body_x, right_body_y), 10, (0, 255, 0), -1)

    elif right_body_angle >= 0.5 and right_body_angle < 20:
        right_body_point = 2 + right_body_plus
        cv2.circle(output_image, (right_body_x, right_body_y), 10, (0, 255, 255), -1)

    elif right_body_angle >= 20 and right_body_angle < 60:
        right_body_point = 3 + right_body_plus
        cv2.circle(output_image, (right_body_x, right_body_y), 10, (0, 165, 255), -1)

    elif right_body_angle >= 60:
        right_body_point = 4 + right_body_plus
        cv2.circle(output_image, (right_body_x, right_body_y), 10, (0, 0, 255), -1)



    global right_body_point_live
    right_body_point_live.append(right_body_point)
    right_body_point_avg = (sum(right_body_point_live) / len(right_body_point_live))
    right_body_point_avg = round(right_body_point_avg, 1)

        # Power, Muscle Point 힘, 무게 점수
    # ----------------------------------------------------------------------------------------------------------------
    muscle_point = 1
    power_point = 1


    # 그룹 A 점수화 행렬 연산
    right_groupA_point = np.array(
        [
            [
                [[1, 2, 2, 3], [2, 2, 3, 3], [2, 3, 3, 4]],
                [[2, 2, 3, 3], [2, 2, 3, 3], [3, 3, 3, 4]]
            ],

            [
                [[2, 3, 3, 4], [3, 3, 3, 4], [3, 4, 4, 5]],
                [[3, 3, 4, 4], [3, 3, 4, 4], [4, 4, 4, 5]]
            ],

            [
                [[3, 4, 4, 5], [3, 4, 4, 5], [4, 4, 4, 5]],
                [[3, 4, 4, 5], [4, 4, 4, 5], [4, 4, 5, 5]]
            ],

            [
                [[4, 4, 4, 5], [4, 4, 4, 5], [4, 4, 5, 6]],
                [[4, 4, 5, 5], [4, 4, 5, 5], [4, 4, 5, 6]]
            ],

            [
                [[5, 5, 5, 6], [5, 6, 6, 7], [6, 6, 7, 7]],
                [[5, 5, 6, 7], [6, 6, 7, 7], [6, 7, 7, 8]]
            ],

            [
                [[7, 7, 7, 8], [8, 8, 8, 9], [9, 9, 9, 9]],
                [[7, 7, 8, 9], [8, 8, 9, 9], [9, 9, 9, 9]]
            ]
        ])

    right_Score_A = right_groupA_point[right_upper_point - 1][right_wrist_twist - 1][right_lower_point - 1][
        right_wrist_point - 1]


    # 그룹 B 점수화 행렬 연산
    right_groupB_point = np.array(
        [
            [
                [1, 2, 3, 5, 6, 7], [2, 2, 4, 5, 6, 7], [3, 3, 4, 5, 6, 7], [5, 5, 6, 7, 7, 8], [7, 7, 7, 8, 8, 8],
                [8, 8, 8, 8, 9, 9]
            ],
            [
                [3, 3, 4, 5, 6, 7], [3, 3, 5, 5, 7, 7], [3, 4, 5, 6, 7, 7], [5, 6, 7, 7, 7, 8], [7, 7, 8, 8, 8, 8],
                [8, 8, 8, 9, 9, 9]
            ]
        ])

    right_Score_B = right_groupB_point[right_leg_point - 1][right_neck_point - 1][right_body_point - 1]

    # 그룹 C, D 점수화
    right_Score_C = right_Score_A + muscle_point + power_point

    right_Score_D = right_Score_B + muscle_point + power_point

    # 최종 점수 점수화
    Grand = np.array(
        [[1, 2, 3, 4, 4, 5, 5], [2, 2, 3, 4, 4, 5, 5], [3, 3, 3, 4, 4, 4, 6], [3, 3, 3, 4, 5, 6, 6],
         [4, 4, 4, 5, 6, 7, 7], [4, 4, 5, 6, 6, 7, 7], [5, 5, 6, 6, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7]])

    if right_Score_C > 8:
        if right_Score_D > 7:
            right_Score_Grand = Grand[7][6]

        else:
            right_Score_Grand = Grand[7][right_Score_D - 1]

    elif right_Score_C <= 8:
        if right_Score_D > 7:
            right_Score_Grand = Grand[right_Score_C - 1][6]

        else:
            right_Score_Grand = Grand[right_Score_C - 1][right_Score_D - 1]

    global right_grand_point_live
    right_grand_point_live.append(right_Score_Grand)
    right_grand_point_avg = (sum(right_grand_point_live) / len(right_grand_point_live))
    right_grand_point_avg = round(right_grand_point_avg, 1)

## 실시간 점수
    cv2.putText(output_image, "Upper arm score : " + str(right_upper_point), (10, 450), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 255, 0), 2)
    cv2.putText(output_image, "Lower arm score : " + str(right_lower_point), (10, 480), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 255, 0), 2)
    cv2.putText(output_image, "Wrist score : " + str(right_wrist_point), (10, 510), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0),
                2)

    cv2.putText(output_image, "Neck score : " + str(right_neck_point), (10, 540), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.putText(output_image, "Body score : " + str(right_body_point), (10, 570), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0),
                2)
    cv2.putText(output_image, "Leg score : " + str(right_leg_point), (10, 600), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.putText(output_image, "Grand Score : " + str(right_Score_Grand), (10, 630), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

## 평균 점수
    cv2.putText(output_image, "Upper arm avg : " + str(right_upper_point_avg), (850, 450), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Lower arm avg : " + str(right_lower_point_avg), (850, 480), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Wrist avg : " + str(right_wrist_point_avg), (850, 510), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Neck avg : " + str(right_neck_point_avg), (850, 540), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Body avg : " + str(right_body_point_avg), (850, 570), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Leg avg : " + str(right_leg_point), (850, 600), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Grand score avg : " + str(right_grand_point_avg), (850, 630), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    return output_image


def classifyPose_left(landmarks, output_image):

    global left_lower_point
    global left_upper_point
    global left_wrist_point
    global left_neck_point
    global left_body_point
    global left_leg_point
    global left_Score_A, left_Score_B, left_Score_Grand


    left_elbow_angle = -1 * (calculateAngle_left(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]))

    left_shoulder_angle = -1 * ((calculateAngle_left(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])) - 180)

    left_body = -1 * (calculateAngle((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0],
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1] + 200,
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][2]),
                                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) + 180)  # 왼쪽 몸통 각도

    left_wrist = calculateAngle_left(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value])  # 왼쪽 손목 각도


    left_neck = -1 * (calculateAngle((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0],
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1] + 200,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][2]),
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                               landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]) + 190)  # 왼쪽 목 각도

    """RULA"""

    # Upper arm 상완
    left_upper_angle = round(left_shoulder_angle, 1)  # 왼쪽 상완, 소숫점 두번째 자리에서 반올림

    # Lower arm 전완
    left_lower_angle = round(left_elbow_angle, 1)  # 왼쪽 전완, 소숫점 두번째 자리에서 반올림

    # Wrist 손목
    left_wrist_angle = round(left_wrist, 1)  # 왼쪽 손목, 소숫점 두번째 자리에서 반올림

    # Neck 목
    left_neck_angle = round(left_neck, 1)  #

    # body 몸통
    left_body_angle = round(left_body, 1)  # 왼쪽 몸통, 소숫점 두번째 자리에서 반올림

    # Upper Arm Point 상완 점수
    # ----------------------------------------------------------------------------------------------------------------

    left_upper_x = int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] + landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0]) / 2)
    left_upper_y = int((landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] + landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1]) / 2)

    if left_upper_angle > -10 and left_upper_angle < 20 :
        left_upper_point = 1
        cv2.circle(output_image, (left_upper_x, left_upper_y), 10, (0, 255, 0), -1)

    elif left_upper_angle > 20 and left_upper_angle < 45 :
        left_upper_point = 2
        cv2.circle(output_image, (left_upper_x, left_upper_y), 10, (0, 255, 255), -1)

    elif left_upper_angle > 45 and left_upper_angle < 90 :
        left_upper_point = 3
        cv2.circle(output_image, (left_upper_x, left_upper_y), 10, (0, 165, 255), -1)

    else :
        left_upper_point = 4
        cv2.circle(output_image, (left_upper_x, left_upper_y), 10, (0, 0, 255), -1)

    global left_upper_point_live
    left_upper_point_live.append(left_upper_point)
    left_upper_point_avg = (sum(left_upper_point_live) / len(left_upper_point_live))
    left_upper_point_avg = round(left_upper_point_avg, 1)

    # Lower Arm Point 전완 점수
    # ----------------------------------------------------------------------------------------------------------------

    left_lower_x = int((landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0] + landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0]) / 2)
    left_lower_y = int((landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1] + landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]) / 2)

    if left_lower_angle > 60 and left_lower_angle < 100 :
        left_lower_point = 1
        cv2.circle(output_image, (left_lower_x, left_lower_y), 10, (0, 255, 0), -1)

    else:
        left_lower_point = 2
        cv2.circle(output_image, (left_lower_x, left_lower_y), 10, (0, 0, 255), -1)

    global left_lower_point_live
    left_lower_point_live.append(left_lower_point)
    left_lower_point_avg = (sum(left_lower_point_live) / len(left_lower_point_live))
    left_lower_point_avg = round(left_lower_point_avg, 1)

    # Wrist Point 손목 점수
    # ----------------------------------------------------------------------------------------------------------------

    left_wrist_twist = 1

    if left_wrist_angle > -1 and left_wrist_angle < 1:
        left_wrist_point = 1
        cv2.circle(output_image, (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0],
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]), 10, (0, 255, 0), -1)

    elif (left_wrist_angle < -1 and left_wrist_angle > -15) or (left_wrist_angle > 1 and left_wrist_angle < 15):
        left_wrist_point = 2
        cv2.circle(output_image, (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0],
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]), 10, (0, 255, 255), -1)

    elif left_wrist_angle > 15 or left_wrist_angle < -15:
        left_wrist_point = 3
        cv2.circle(output_image, (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0],
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]), 10, (0, 0, 255), -1)

    global left_wrist_point_live
    left_wrist_point_live.append(left_wrist_point)
    left_wrist_point_avg = (sum(left_wrist_point_live) / len(left_wrist_point_live))
    left_wrist_point_avg = round(left_wrist_point_avg, 1)

    # Neck Point 목 점수
    # ----------------------------------------------------------------------------------------------------------------

    left_neck_x = int((landmarks[mp_pose.PoseLandmark.LEFT_EAR.value][0] + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0]) / 2)
    left_neck_y = int((landmarks[mp_pose.PoseLandmark.LEFT_EAR.value][1] + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]) / 2)

    if left_neck_angle > -3 and left_neck_angle < 10:
        left_neck_point = 1
        cv2.circle(output_image, (left_neck_x, left_neck_y), 10, (0, 255, 0), -1)

    elif left_neck_angle > 10 and left_neck_angle < 20:
        left_neck_point = 2
        cv2.circle(output_image, (left_neck_x, left_neck_y), 10, (0, 255, 255), -1)

    elif left_neck_angle > 20:
        left_neck_point = 3
        cv2.circle(output_image, (left_neck_x, left_neck_y), 10, (0, 165, 255), -1)

    elif left_neck_angle < -3:
        left_neck_point = 4
        cv2.circle(output_image, (left_neck_x, left_neck_y), 10, (0, 0, 255), -1)

    global left_neck_point_live
    left_neck_point_live.append(left_neck_point)
    left_neck_point_avg = (sum(left_neck_point_live) / len(left_neck_point_live))
    left_neck_point_avg = round(left_neck_point_avg, 1)

        # Body Point 몸통 점수
    # ----------------------------------------------------------------------------------------------------------------

    left_body_x = int((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0]) / 2)
    left_body_y = int((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1] + landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]) / 2)

    if left_body_angle > -5 and left_body_angle < 0.05:
        left_body_point = 1
        cv2.circle(output_image, (left_body_x, left_body_y), 10, (0, 255, 0), -1)

    elif left_body_angle > 0.05 and left_body_angle < 20:
        left_body_point = 2
        cv2.circle(output_image, (left_body_x, left_body_y), 10, (0, 255, 255), -1)

    elif left_body_angle > 20 and left_body_angle < 60:
        left_body_point = 3
        cv2.circle(output_image, (left_body_x, left_body_y), 10, (0, 165, 255), -1)

    elif left_body_angle > 60:
        left_body_point = 4
        cv2.circle(output_image, (left_body_x, left_body_y), 10, (0, 0, 255), -1)

    global left_body_point_live
    left_body_point_live.append(left_body_point)
    left_body_point_avg = (sum(left_body_point_live) / len(left_body_point_live))
    left_body_point_avg = round(left_body_point_avg, 1)

        # Power, Muscle Point 힘, 무게 점수
    # ----------------------------------------------------------------------------------------------------------------
    muscle_point = 1
    power_point = 1


    # 그룹 A 점수화 행렬 연산
    left_groupA_point = np.array(
        [
            [
                [[1, 2, 2, 3], [2, 2, 3, 3], [2, 3, 3, 4]],
                [[2, 2, 3, 3], [2, 2, 3, 3], [3, 3, 3, 4]]
            ],

            [
                [[2, 3, 3, 4], [3, 3, 3, 4], [3, 4, 4, 5]],
                [[3, 3, 4, 4], [3, 3, 4, 4], [4, 4, 4, 5]]
            ],

            [
                [[3, 4, 4, 5], [3, 4, 4, 5], [4, 4, 4, 5]],
                [[3, 4, 4, 5], [4, 4, 4, 5], [4, 4, 5, 5]]
            ],

            [
                [[4, 4, 4, 5], [4, 4, 4, 5], [4, 4, 5, 6]],
                [[4, 4, 5, 5], [4, 4, 5, 5], [4, 4, 5, 6]]
            ],

            [
                [[5, 5, 5, 6], [5, 6, 6, 7], [6, 6, 7, 7]],
                [[5, 5, 6, 7], [6, 6, 7, 7], [6, 7, 7, 8]]
            ],

            [
                [[7, 7, 7, 8], [8, 8, 8, 9], [9, 9, 9, 9]],
                [[7, 7, 8, 9], [8, 8, 9, 9], [9, 9, 9, 9]]
            ]
        ])

    left_Score_A = left_groupA_point[left_upper_point - 1][left_wrist_twist - 1][left_lower_point - 1][
        left_wrist_point - 1]

    # 그룹 B 점수화 행렬 연산
    left_groupB_point = np.array(
        [
            [
                [1, 2, 3, 5, 6, 7], [2, 2, 4, 5, 6, 7], [3, 3, 4, 5, 6, 7], [5, 5, 6, 7, 7, 8], [7, 7, 7, 8, 8, 8],
                [8, 8, 8, 8, 9, 9]
            ],
            [
                [3, 3, 4, 5, 6, 7], [3, 3, 5, 5, 7, 7], [3, 4, 5, 6, 7, 7], [5, 6, 7, 7, 7, 8], [7, 7, 8, 8, 8, 8],
                [8, 8, 8, 9, 9, 9]
            ]
        ])

    left_Score_B = left_groupB_point[left_leg_point - 1][left_neck_point - 1][left_body_point - 1]

    # 그룹 C, D 점수화
    left_Score_C = left_Score_A + muscle_point + power_point

    left_Score_D = left_Score_B + muscle_point + power_point

    # 최종 점수 점수화
    Grand = np.array(
        [[1, 2, 3, 4, 4, 5, 5], [2, 2, 3, 4, 4, 5, 5], [3, 3, 3, 4, 4, 4, 6], [3, 3, 3, 4, 5, 6, 6],
         [4, 4, 4, 5, 6, 7, 7], [4, 4, 5, 6, 6, 7, 7], [5, 5, 6, 6, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7]])


    if left_Score_C > 8:
        if left_Score_D > 7:
            left_Score_Grand = Grand[7][6]

        else:
            left_Score_Grand = Grand[7][left_Score_D - 1]


    elif left_Score_C <= 8:
        if left_Score_D > 7:
            left_Score_Grand = Grand[left_Score_C - 1][6]

        else:
            left_Score_Grand = Grand[left_Score_C - 1][left_Score_D - 1]

    global left_grand_point_live
    left_grand_point_live.append(left_Score_Grand)
    left_grand_point_avg = (sum(left_grand_point_live) / len(left_grand_point_live))
    left_grand_point_avg = round(left_grand_point_avg, 1)

## 실시간 점수
    cv2.putText(output_image, "Upper arm score : " + str(left_upper_point), (10, 450), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 255, 0), 2)
    cv2.putText(output_image, "Lower arm score : " + str(left_lower_point), (10, 480), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 255, 0), 2)
    cv2.putText(output_image, "Wrist score : " + str(left_wrist_point), (10, 510), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0),
                2)

    cv2.putText(output_image, "Neck score : " + str(left_neck_point), (10, 540), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.putText(output_image, "Body score : " + str(left_body_point), (10, 570), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0),
                2)
    cv2.putText(output_image, "Leg score : " + str(left_leg_point), (10, 600), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.putText(output_image, "Grand Score : " + str(left_Score_Grand), (10, 630), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

## 평균 점수
    cv2.putText(output_image, "Upper arm avg : " + str(left_upper_point_avg), (850, 450), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Lower arm avg : " + str(left_lower_point_avg), (850, 480), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Wrist avg : " + str(left_wrist_point_avg), (850, 510), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Neck avg : " + str(left_neck_point_avg), (850, 540), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Body avg : " + str(left_body_point_avg), (850, 570), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Leg avg : " + str(left_leg_point), (850, 600), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)

    cv2.putText(output_image, "Grand score avg : " + str(left_grand_point_avg), (850, 630), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 0, 255), 2)


    return output_image


#=======================================================================================================================
def loadWebCam_right():
    global right_lower_point
    global right_upper_point
    global right_wrist_point
    global right_neck_point
    global right_body_point
    global right_leg_point



    global right_Score_A, right_Score_B, right_Score_Grand


    #right visualization
    groupA_point_right = open('C:/Users/{}/Desktop/My Program/Right/Point/groupA_point_right.csv'.format(os.getlogin()), 'w', newline='')
    groupB_point_right = open('C:/Users/{}/Desktop/My Program/Right/Point/groupB_point_right.csv'.format(os.getlogin()), 'w', newline='')
    grand_point_right = open('C:/Users/{}/Desktop/My Program/Right/Point/grand_point_right.csv'.format(os.getlogin()), 'w', newline='')

    groupA_point_right_csv = csv.writer(groupA_point_right)
    R_groupA_point_right = ['right_upper_point', 'right_lower_point', 'right_wrist_point']
    groupA_point_right_csv.writerow(R_groupA_point_right)

    groupB_point_right_csv = csv.writer(groupB_point_right)
    R_groupB_point_right = ['right_body_point', 'right_neck_point']
    groupB_point_right_csv.writerow(R_groupB_point_right)

    grand_point_right_csv = csv.writer(grand_point_right)
    R_grand_point_right = ['right_grand_point']
    grand_point_right_csv.writerow(R_grand_point_right)

    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    camera_video = cv2.VideoCapture(1)
    camera_video.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    camera_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start = time.time() # 시작 시간 저장

    start_now = datetime.now()
    print("시작시각은 : ", start_now.strftime('%Y-%m-%d %H:%M:%S'), "입니다.")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 영상을 기록할 코덱 설정
    is_record = False  # 녹화상태는 처음엔 거짓으로 설정
    is_capture = False

    # Initialize a resizable window.
    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pose Classification', 1080, 720)

    windows_user_name = os.path.expanduser('~')


    while camera_video.isOpened():

        now = datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        nowDatetime_path = now.strftime('%Y-%m-%d %H_%M_%S')

        ok, frame = camera_video.read()

        if not ok:
            continue

        cv2.putText(frame, text=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

        frame_height, frame_width, _ = frame.shape

        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        frame, landmarks = detectPose(frame, pose_video)

        if landmarks:
            frame = classifyPose_right(landmarks, frame)

        ################################################ 추가내용 ############################################
        R_groupA_point_right = [right_upper_point, right_lower_point, right_wrist_point]
        groupA_point_right_csv.writerow(R_groupA_point_right)

        R_groupB_point_right = [right_body_point, right_neck_point]
        groupB_point_right_csv.writerow(R_groupB_point_right)

        R_grand_point_right = [right_Score_Grand]
        grand_point_right_csv.writerow(R_grand_point_right)
        #######################################################################################################

        key = cv2.waitKey(1) & 0xFF

        ################################ For Visualization 가즈아ㅏㅏㅏ#############################
        '''
        img = cv2.imread("bodypoint_R.jpg") # 이미지 불러오기
        resize_img = cv2.resize(img, (480, 720), interpolation=cv2.INTER_CUBIC) # 사이즈 재조정~
        '''
        img = cv2.imdecode(R_image_nparray, cv2.IMREAD_COLOR)
        resize_img = cv2.resize(img, (480, 720), interpolation=cv2.INTER_CUBIC)  # 사이즈 재조정~

        global right_upper_count, right_lower_count, right_wrist_count, right_body_count, right_neck_count

        if right_upper_point == 3 or right_upper_point == 4:
            right_upper_count += 1

        if right_upper_count <= 280:
            cv2.circle(resize_img, (162, 250), int(1 + (right_upper_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif right_upper_count > 280:
            cv2.circle(resize_img, (162, 250), 15, (0, 0, 255), -1) # 노가다 원(빨간원)


        if right_lower_point == 3:
            right_lower_count += 1

        if right_lower_count <= 280:
            cv2.circle(resize_img, (140, 330), int(1 + (right_lower_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif right_lower_count > 280:
            cv2.circle(resize_img, (140, 330), 15, (0, 0, 255), -1) # 노가다 원(빨간원)


        if right_wrist_point == 3 or right_wrist_point == 4:
            right_wrist_count += 1

        if right_wrist_count <= 280:
            cv2.circle(resize_img, (126, 370), int(1 + (right_wrist_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif right_wrist_count > 280:
            cv2.circle(resize_img, (126, 370), 15, (0, 0, 255), -1) # 노가다 원(빨간원)


        if right_neck_point == 3 or right_neck_point == 4:
            right_neck_count += 1

        if right_neck_count <= 280:
            cv2.circle(resize_img, (240, 150), int(1 + (right_neck_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif right_neck_count > 280:
            cv2.circle(resize_img, (240, 150), 15, (0, 0, 255), -1) # 노가다 원(빨간원)


        if right_body_point == 3 or right_body_point == 4:
            right_body_count += 1

        if right_body_count <= 280:
            cv2.circle(resize_img, (240, 220), int(1 + (right_body_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif right_body_count > 280:
            cv2.circle(resize_img, (240, 220), 15, (0, 0, 255), -1) # 노가다 원(빨간원)
                # cv2.circle(resize_img, (240, 220), 15, (0, 0, 255), -1) # 노가다 원(빨간원)

            # legs
        if right_leg_point == 1:
            cv2.circle(resize_img, (270, 560), 3, (0, 255, 0), -1) # 노가다 원(초록원)

        elif right_leg_point == 2:
            cv2.circle(resize_img, (270, 560), 15, (0, 0, 255), -1) # 노가다 원(빨간원)


        cv2.imshow("BodyPoint", resize_img) # 출력
        # cv2.setMouseCallback("BodyPoint", onMouse_RB1)
        call_func()

        if key == ord('r') and is_record == False:
            is_record = True
            video = cv2.VideoWriter("C:/Users/{}/Desktop/My Program/Right/Record/".format(os.getlogin()) + nowDatetime_path + ".mp4", fourcc, 10,
                                        (frame.shape[1], frame.shape[0]))
        elif key == ord('r') and is_record == True:
            is_record = False
            video.release()

        elif key == ord('c'):
            is_capture = True
            start_time = time.time()
            capture_ext = ".png"

            im1 = frame
            im2 = resize_img

            def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
                h_min = min(im.shape[0] for im in im_list)
                im_list_resize = [
                    cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                    for im in im_list]
                return cv2.hconcat(im_list_resize)

            im_h_resize = hconcat_resize_min([im1, im2])
            cv2.imwrite("C:/Users/{}/Desktop/My Program/Right/Captured/{}{}".format(os.getlogin(), nowDatetime_path, capture_ext), im_h_resize)

        elif key == ord('q'):
            break

        elif (key == 27):
            groupA_point_right.close()
            groupB_point_right.close()
            grand_point_right.close()
            folder_path = "C:/Users/{}/Desktop".format(os.getlogin())
            folder_path = os.path.realpath(folder_path)
            os.startfile(folder_path)
            break


        if is_record == True:
            video.write(frame)
            cv2.circle(img=frame, center=(620, 15), radius=5, color=(0, 0, 255), thickness=-1)

        if is_capture == True:
            cv2.putText(frame, "Captured", org=(900, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(0, 255, 0), thickness=2)
            if time.time() - start_time > 2:
                is_capture = False

        ### 오른쪽 자동캡쳐
        global img_counter_right_upper
        global img_counter_right_lower
        global img_counter_right_wrist
        global img_counter_right_neck
        global img_counter_right_body
        global img_counter_right_grand_score
        global right_upper_capture_time, right_upper_capture_time2
        global right_lower_capture_time, right_lower_capture_time2
        global right_wrist_capture_time, right_wrist_capture_time2
        global right_body_capture_time, right_body_capture_time2
        global right_neck_capture_time, right_neck_capture_time2
        global right_grand_capture_time, right_grand_capture_time2

        if right_upper_point > 3:

            right_upper_capture_time = time.time()

            if abs(right_upper_capture_time2 - right_upper_capture_time) > 1:
                img_name_right_upper = "right_upper{}.png".format(img_counter_right_upper)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/right_upper/'.format(os.getlogin()) + img_name_right_upper,
                            frame)
                print("{} written!".format(img_name_right_upper))
                img_counter_right_upper += 1
                right_upper_capture_time2 = time.time()

        if right_lower_point > 1:

            right_lower_capture_time = time.time()

            if abs(right_lower_capture_time2 - right_lower_capture_time) > 1:
                img_name_right_lower = "right_lower{}.png".format(img_counter_right_lower)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/right_lower/'.format(os.getlogin()) + img_name_right_lower,
                            frame)
                print("{} written!".format(img_name_right_lower))
                img_counter_right_lower += 1
                right_lower_capture_time2 = time.time()

        if right_wrist_point > 2:

            right_wrist_capture_time = time.time()

            if abs(right_wrist_capture_time2 - right_wrist_capture_time) > 1:
                img_name_right_wrist = "right_wrist{}.png".format(img_counter_right_wrist)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/right_wrist/'.format(os.getlogin()) + img_name_right_wrist,
                            frame)
                print("{} written!".format(img_name_right_wrist))
                img_counter_right_wrist += 1
                right_wrist_capture_time2 = time.time()

        if right_neck_point > 2:

            right_neck_capture_time = time.time()

            if abs(right_neck_capture_time2 - right_neck_capture_time) > 1:
                img_name_right_neck = "right_neck{}.png".format(img_counter_right_neck)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/right_neck/'.format(os.getlogin()) + img_name_right_neck,
                            frame)
                print("{} written!".format(img_name_right_neck))
                img_counter_right_neck += 1
                right_neck_capture_time2 = time.time()

        if right_body_point > 2:

            right_body_capture_time = time.time()

            if abs(right_body_capture_time2 - right_body_capture_time) > 1:
                img_name_right_body = "right_body{}.png".format(img_counter_right_body)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/right_body/'.format(os.getlogin()) + img_name_right_body,
                            frame)
                print("{} written!".format(img_name_right_body))
                img_counter_right_body += 1
                right_body_capture_time2 = time.time()

        if right_Score_Grand > 2:

            right_grand_capture_time = time.time()

            if abs(right_grand_capture_time2 - right_grand_capture_time) > 1:
                img_name_right_grand_score = "right_grand_score{}.png".format(img_counter_right_grand_score)
                cv2.imwrite(
                    'C:/Users/{}/Desktop/My Program/Right/Captured/right_grand_score/'.format(os.getlogin()) + img_name_right_grand_score,
                    frame)
                print("{} written!".format(img_name_right_grand_score))
                img_counter_right_grand_score += 1
                right_grand_capture_time2 = time.time()

        running_time = math.trunc(time.time() - start)

        cv2.putText(frame, "Time : " + str(running_time), (230, 70), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 255, 0), 2)

        cv2.imshow("Pose Classification", frame)


    camera_video.release()

    cv2.destroyAllWindows()

    ################################ 추가내용 (프로그램 작동시간) #############################
    work_time = int(time.time() - start)
    print("time:", work_time) # 현재시각 - 시작시간 = 실행시간

    end_now = datetime.now()
    print("끝나는 시각은 : ", end_now.strftime('%Y-%m-%d %H:%M:%S'), "입니다.")


    #################################### 추가내용 Data analysis ###############################
    # 데이터프레임 가져오기
    data_groupA_point_right = pd.read_csv('C:/Users/{}/Desktop/My Program/Right/Point/groupA_point_right.csv'.format(os.getlogin()))
    data_groupB_point_right = pd.read_csv('C:/Users/{}/Desktop/My Program/Right/Point/groupB_point_right.csv'.format(os.getlogin()))
    data_grand_point_right = pd.read_csv('C:/Users/{}/Desktop/My Program/Right/Point/grand_point_right.csv'.format(os.getlogin()))


    # 데이터의 갯수
    data_groupA_numbers_right = data_groupA_point_right['right_upper_point'].count()
    print(data_groupA_numbers_right)
    data_groupB_numbers_right = data_groupB_point_right['right_body_point'].count()
    print(data_groupB_numbers_right)
    print(data_grand_point_right['right_grand_point'].count())
    # groupA랑 groupB의 데이터 갯수는 똑같음


    # 시간을 포함한 데이터프레임 완성
    number_per_sec_right = round(data_groupA_numbers_right/work_time, 1)  # 초당 데이터 갯수

    sec = []
    for x in range(work_time):
        while len(sec) != int(number_per_sec_right*(x + 1)):
                sec.append(x)
    print(sec)
    len(sec)

    sec = pd.Series(sec, name='Time')

    data_groupA_point_right = pd.concat((data_groupA_point_right,sec), axis=1)
    data_groupB_point_right = pd.concat((data_groupB_point_right,sec), axis=1)
    data_grand_point_right = pd.concat((data_grand_point_right,sec), axis=1)

    data_groupA_point_right = data_groupA_point_right.dropna() # 반올림으로 인해 1개까지 차이가 날 수 있음. 그걸 제거
    data_groupB_point_right = data_groupB_point_right.dropna() # 1개 제거하더라도 대세엔 지장이 없음
    data_grand_point_right = data_grand_point_right.dropna()

    data_groupA_point_right = data_groupA_point_right.groupby(by='Time').mean()
    data_groupB_point_right = data_groupB_point_right.groupby(by='Time').mean()
    data_grand_point_right = data_grand_point_right.groupby(by='Time').mean()


    # Visualization
    # 그룹A 그래프
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plt.ylabel('Right groupA Point')
    sns.lineplot(data=data_groupA_point_right, x="Time", y="right_upper_point", label="right_upper")
    sns.lineplot(data=data_groupA_point_right, x="Time", y="right_lower_point", label="right_lower")
    sns.lineplot(data=data_groupA_point_right, x="Time", y="right_wrist_point", label="right_wrist")
    plt.xticks(np.arange(0,work_time,int(work_time/7)))
    plt.savefig('C:/Users/{}/Desktop/My Program/Right/Graph/GroupA_Graph_right.png'.format(os.getlogin()))


    # 그룹B 그래프
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plt.ylabel('Right groupB Point')
    sns.lineplot(data=data_groupB_point_right, x="Time", y="right_body_point", label="right_body")
    sns.lineplot(data=data_groupB_point_right, x="Time", y="right_neck_point", label="right_neck")
    plt.xticks(np.arange(0,work_time,int(work_time/7)))
    plt.savefig('C:/Users/{}/Desktop/My Program/Right/Graph/GroupB_Graph_right.png'.format(os.getlogin()))


    # 최종점수 그래프
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plt.ylabel('Right Grand Score')
    sns.lineplot(data=data_grand_point_right, x="Time", y="right_grand_point", label="right_grand_point")

    plt.xticks(np.arange(0,work_time,int(work_time/7)))
    plt.savefig('C:/Users/{}/Desktop/My Program/Right/Graph/Grand_Score_Graph_right.png'.format(os.getlogin()))

    # fig, axes = plt.subplots(3,1, figsize=(10,10)) ##### 기본 틀이 만들어짐
    #
    # sns.scatterplot(data=data_groupA_point_right, x="Time", y="right_upper_point", ax=axes[0]) ##### ax는 축을 지정 [0,0]은 첫번쨰
    # sns.scatterplot(data=data_groupA_point_right, x="Time", y="right_lower_point", ax=axes[0]) ##### ax는 축을 지정 [0,0]은 첫번쨰
    # sns.scatterplot(data=data_groupA_point_right, x="Time", y="right_wrist_point", ax=axes[0]) ##### ax는 축을 지정 [0,0]은 첫번쨰
    #
    #
    # sns.scatterplot(data=data_groupB_point_right, x="Time", y="right_body_point", ax=axes[1])
    # sns.scatterplot(data=data_groupB_point_right, x="Time", y="right_neck_point", ax=axes[1])
    #
    # sns.scatterplot(data=data_grand_point_right, x="Time", y="right_grand_point", ax=axes[2])
    # plt.show()

    # 데이터 요약
    print("관절별 부하의 정도")
    mean_of_right_upper_point = data_groupA_point_right["right_upper_point"].mean()
    mean_of_right_lower_point = data_groupA_point_right["right_lower_point"].mean()
    mean_of_right_wrist_point = data_groupA_point_right["right_wrist_point"].mean()
    mean_of_right_body_point = data_groupB_point_right["right_body_point"].mean()
    mean_of_right_neck_point = data_groupB_point_right["right_neck_point"].mean()
    print("right_upper_point의 평균부하의 정도 :", mean_of_right_upper_point)
    print("right_lower_point의 평균부하의 정도 :", mean_of_right_lower_point)
    print("right_wrist_point의 평균부하의 정도 :", mean_of_right_wrist_point)
    print("right_body_point의 평균부하의 정도 :", mean_of_right_body_point)
    print("right_neck_point의 평균부하의 정도 :", mean_of_right_neck_point)

#=======================================================================================================

def onMouse_RU1(event, x, y, flags, param):
    global right_upper_plus

    global right_upper_page

    img_Rupper = cv2.imread("C:/Users/{}/Desktop/img_Rupper.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 152 < x < 172 and 240 < y < 260:
        if right_upper_page == 1:
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 2:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 3:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 4:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 5:
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 6:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 7:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 8:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 9:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 10:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 11:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 12:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 13:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 14:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 15:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

        elif right_upper_page == 16:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)

            cv2.setMouseCallback("Upper Consider", onMouse_RU2)

def onMouse_RU2(event, x, y, flags, param):
    global right_upper_plus

    global right_upper_page

    img_Rupper = cv2.imread("C:/Users/{}/Desktop/img_Rupper.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 45 < y < 115:
        if right_upper_page == 1:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 2

        elif right_upper_page == 2:
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 1

        elif right_upper_page == 3:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 2
            right_upper_page = 6

        elif right_upper_page == 4:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 7

        elif right_upper_page == 5:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 8

        elif right_upper_page == 6:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 3

        elif right_upper_page == 7:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 4

        elif right_upper_page == 8:
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 5

        elif right_upper_page == 9:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 12

        elif right_upper_page == 10:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 13

        elif right_upper_page == 11:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 14

        elif right_upper_page == 12:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 9

        elif right_upper_page == 13:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 10

        elif right_upper_page == 14:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -2
            right_upper_page = 11

        elif right_upper_page == 15:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 16

        elif right_upper_page == 16:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 15

    elif event == cv2.EVENT_LBUTTONDOWN and 115 < y < 190:
        if right_upper_page == 1:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 3

        elif right_upper_page == 2:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 2
            right_upper_page = 6

        elif right_upper_page == 3:
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 1

        elif right_upper_page == 4:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 9

        elif right_upper_page == 5:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 10

        elif right_upper_page == 6:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 2

        elif right_upper_page == 7:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 12

        elif right_upper_page == 8:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 13

        elif right_upper_page == 9:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 4

        elif right_upper_page == 10:
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 5

        elif right_upper_page == 11:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 15

        elif right_upper_page == 12:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 7

        elif right_upper_page == 13:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 8

        elif right_upper_page == 14:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 16

        elif right_upper_page == 15:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -2
            right_upper_page = 11

        elif right_upper_page == 16:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 14


    elif event == cv2.EVENT_LBUTTONDOWN and 190 < y < 250:
        if right_upper_page == 1:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 4

        elif right_upper_page == 2:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 7

        elif right_upper_page == 3:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 9

        elif right_upper_page == 4:
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 1

        elif right_upper_page == 5:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -2
            right_upper_page = 11

        elif right_upper_page == 6:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 12

        elif right_upper_page == 7:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 2

        elif right_upper_page == 8:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 14

        elif right_upper_page == 9:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 3

        elif right_upper_page == 10:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 15

        elif right_upper_page == 11:
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 5

        elif right_upper_page == 12:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 2
            right_upper_page = 6

        elif right_upper_page == 13:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 16

        elif right_upper_page == 14:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 8

        elif right_upper_page == 15:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 10

        elif right_upper_page == 16:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 14

    elif event == cv2.EVENT_LBUTTONDOWN and 250 < y < 313:
        if right_upper_page == 1:
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 5

        elif right_upper_page == 2:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 8

        elif right_upper_page == 3:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 10

        elif right_upper_page == 4:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 11

        elif right_upper_page == 5:
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 1

        elif right_upper_page == 6:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 13

        elif right_upper_page == 7:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 14

        elif right_upper_page == 8:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 2

        elif right_upper_page == 9:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 15

        elif right_upper_page == 10:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 3

        elif right_upper_page == 11:
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = -1
            right_upper_page = 4

        elif right_upper_page == 12:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 295), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 16

        elif right_upper_page == 13:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 2
            right_upper_page = 6

        elif right_upper_page == 14:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 7

        elif right_upper_page == 15:
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 0
            right_upper_page = 9

        elif right_upper_page == 16:
            cv2.circle(img_Rupper, (892, 95), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 168), 5, (0, 0, 255), -1)
            cv2.circle(img_Rupper, (892, 235), 5, (0, 0, 255), -1)
            cv2.imshow("Upper Consider", img_Rupper)
            right_upper_plus = 1
            right_upper_page = 12

#====================================================================================
def onMouse_RL1(event, x, y, flags, param):
    global right_lower_plus

    global right_lower_page

    img_Rlower = cv2.imread("C:/Users/{}/Desktop/img_Rlower.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 130 < x < 150 and 320 < y < 340:
        if right_lower_page == 1:
            cv2.imshow("Lower Consider", img_Rlower)
            right_lower_plus = 0

            cv2.setMouseCallback("Lower Consider", onMouse_RL2)

        elif right_lower_page == 2:
            cv2.circle(img_Rlower, (887, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Lower Consider", img_Rlower)

            cv2.setMouseCallback("Lower Consider", onMouse_RL2)

def onMouse_RL2(event, x, y, flags, param):
    global right_lower_plus

    global right_lower_page

    img_Rlower = cv2.imread("C:/Users/{}/Desktop/img_Rlower.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 50 < y < 201:
        if right_lower_page == 1:
            cv2.circle(img_Rlower, (887, 168), 5, (0, 0, 255), -1)
            cv2.imshow("Lower Consider", img_Rlower)
            right_lower_plus = 1
            right_lower_page = 2

        elif right_lower_page == 2:
            cv2.imshow("Lower Consider", img_Rlower)
            right_lower_plus = 0
            right_lower_page = 1

##############################################################################
def onMouse_RN1(event, x, y, flags, param):
    global right_neck_plus

    global right_neck_page

    img_Rneck = cv2.imread("C:/Users/{}/Desktop/img_Rneck.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 210 < x < 270 and 120 < y < 180:
        if right_neck_page == 1:
            cv2.imshow("Neck Consider", img_Rneck)
            right_neck_plus = 0

            cv2.setMouseCallback("Neck Consider", onMouse_RN2)

        elif right_neck_page == 2:
            cv2.circle(img_Rneck, (887, 102), 5, (0, 0, 255), -1)
            cv2.imshow("Neck Consider", img_Rneck)

            cv2.setMouseCallback("Neck Consider", onMouse_RN2)

        elif right_neck_page == 3:
            cv2.circle(img_Rneck, (887, 185), 5, (0, 0, 255), -1)
            cv2.imshow("Neck Consider", img_Rneck)

            cv2.setMouseCallback("Neck Consider", onMouse_RN2)

        elif right_neck_page == 4:
            cv2.circle(img_Rneck, (887, 102), 5, (0, 0, 255), -1)
            cv2.circle(img_Rneck, (887, 185), 5, (0, 0, 255), -1)
            cv2.imshow("Neck Consider", img_Rneck)

            cv2.setMouseCallback("Neck Consider", onMouse_RN2)

def onMouse_RN2(event, x, y, flags, param):
    global right_neck_plus

    global right_neck_page

    img_Rneck = cv2.imread("C:/Users/{}/Desktop/img_Rneck.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 45 < y < 127:
        if right_neck_page == 1:
            cv2.circle(img_Rneck, (887, 102), 5, (0, 0, 255), -1)
            cv2.imshow("Neck Consider", img_Rneck)
            right_neck_plus = 1
            right_neck_page = 2

        elif right_neck_page == 2:
            cv2.imshow("Neck Consider", img_Rneck)
            right_neck_plus = 0
            right_neck_page = 1

        elif right_neck_page == 3:
            cv2.circle(img_Rneck, (887, 185), 5, (0, 0, 255), -1)
            cv2.circle(img_Rneck, (887, 102), 5, (0, 0, 255), -1)
            cv2.imshow("Neck Consider", img_Rneck)
            right_neck_plus = 2
            right_neck_page = 4

        elif right_neck_page == 4:
            cv2.circle(img_Rneck, (887, 185), 5, (0, 0, 255), -1)
            cv2.imshow("Neck Consider", img_Rneck)
            right_neck_plus = 1
            right_neck_page = 3

    elif event == cv2.EVENT_LBUTTONDOWN and 127 < y < 212:
        if right_neck_page == 1:
            cv2.circle(img_Rneck, (887, 185), 5, (0, 0, 255), -1)
            cv2.imshow("Neck Consider", img_Rneck)
            right_neck_plus = 1
            right_neck_page = 3

        elif right_neck_page == 2:
            cv2.circle(img_Rneck, (887, 185), 5, (0, 0, 255), -1)
            cv2.circle(img_Rneck, (887, 102), 5, (0, 0, 255), -1)
            cv2.imshow("Neck Consider", img_Rneck)
            right_neck_plus = 2
            right_neck_page = 4

        elif right_neck_page == 3:
            cv2.imshow("Neck Consider", img_Rneck)
            right_neck_plus = 0
            right_neck_page = 1

        elif right_neck_page == 4:
            cv2.circle(img_Rneck, (887, 100), 5, (0, 0, 255), -1)
            cv2.imshow("Neck Consider", img_Rneck)
            right_neck_plus = 1
            right_neck_page = 2
#=========================================================================================
def onMouse_RW_and_RWT1(event, x, y, flags, param):
    global right_wrist_plus
    global right_twist_point

    global right_wrist_page

    img_Rwrist_and_twist_and_twist = cv2.imread("C:/Users/{}/Desktop/img_Rwrist_and_twist.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 116 < x < 136 and 360 < y < 380:
        if right_wrist_page == 1:
            cv2.imshow("Wrist Consider", img_Rwrist_and_twist_and_twist)
            right_wrist_plus = 0
            right_twist_point = 1
            cv2.setMouseCallback("Wrist Consider", onMouse_RW_and_RWT2)

        elif right_wrist_page == 2:
            cv2.circle(img_Rwrist_and_twist_and_twist, (887, 130), 5, (0, 0, 255), -1)
            cv2.imshow("Wrist Consider", img_Rwrist_and_twist_and_twist)

            cv2.setMouseCallback("Wrist Consider", onMouse_RW_and_RWT2)

        elif right_wrist_page == 3:
            cv2.circle(img_Rwrist_and_twist_and_twist, (887, 219), 5, (0, 0, 255), -1)
            cv2.imshow("Wrist Consider", img_Rwrist_and_twist_and_twist)

            cv2.setMouseCallback("Wrist Consider", onMouse_RW_and_RWT2)

        elif right_wrist_page == 4:
            cv2.circle(img_Rwrist_and_twist_and_twist, (887, 130), 5, (0, 0, 255), -1)
            cv2.circle(img_Rwrist_and_twist_and_twist, (887, 219), 5, (0, 0, 255), -1)
            cv2.imshow("Wrist Consider", img_Rwrist_and_twist_and_twist)

            cv2.setMouseCallback("Wrist Consider", onMouse_RW_and_RWT2)


def onMouse_RW_and_RWT2(event, x, y, flags, param):
    global right_wrist_plus
    global right_twist_point

    global right_wrist_page


    img_Rwrist_and_twist = cv2.imread("C:/Users/{}/Desktop/img_Rwrist_and_twist.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 45 < y < 140:
        if right_wrist_page == 1:
            cv2.circle(img_Rwrist_and_twist, (887, 124), 5, (0, 0, 255), -1)
            cv2.imshow("Wrist Consider", img_Rwrist_and_twist)
            right_wrist_plus = 1
            right_twist_point = 1
            right_wrist_page = 2

        elif right_wrist_page == 2:
            cv2.circle(img_Rwrist_and_twist, (887, 124), 5, (0, 0, 255), -1)
            cv2.imshow("Wrist Consider", img_Rwrist_and_twist)
            right_wrist_plus = 0
            right_twist_point = 1
            right_wrist_page = 1

        elif right_wrist_page == 3:
            cv2.circle(img_Rwrist_and_twist, (887, 124), 5, (0, 0, 255), -1)
            cv2.circle(img_Rwrist_and_twist, (887, 219), 5, (0, 0, 255), -1)
            right_wrist_plus = 1
            right_twist_point = 2
            right_wrist_page = 4

        elif right_wrist_page == 4:
            cv2.circle(img_Rwrist_and_twist, (887, 124), 5, (0, 0, 255), -1)
            right_wrist_plus = 1
            right_twist_point = 1
            right_wrist_page = 3

    elif event == cv2.EVENT_LBUTTONDOWN and 140 < y < 239:
        if right_wrist_page == 1:
            cv2.circle(img_Rwrist_and_twist, (887, 219), 5, (0, 0, 255), -1)
            cv2.imshow("Wrist Consider", img_Rwrist_and_twist)
            right_wrist_plus = 0
            right_twist_point = 2
            right_wrist_page = 3

        elif right_wrist_page == 2:
            cv2.circle(img_Rwrist_and_twist, (887, 124), 5, (0, 0, 255), -1)
            cv2.circle(img_Rwrist_and_twist, (887, 219), 5, (0, 0, 255), -1)
            cv2.imshow("Wrist Consider", img_Rwrist_and_twist)
            right_wrist_plus = 1
            right_twist_point = 2
            right_wrist_page = 4

        elif right_wrist_page == 3:
            cv2.imshow("Wrist Consider", img_Rwrist_and_twist)
            right_wrist_plus = 0
            right_twist_point = 1
            right_wrist_page = 1

        elif right_wrist_page == 4:
            cv2.circle(img_Rwrist_and_twist, (887, 124), 5, (0, 0, 255), -1)
            cv2.imshow("Wrist Consider")
            right_wrist_plus = 1
            right_twist_point = 1
            right_wrist_page = 2
#============================================================================================
def onMouse_RB1(event, x, y, flags, param):
    global right_body_plus

    global right_body_page

    img_Rbody = cv2.imread("C:/Users/{}/Desktop/img_Rbody.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 220 < x < 260 and 200 < y < 240:
        if right_body_page == 1:
            cv2.imshow("Body Consider", img_Rbody)
            right_body_plus = 0

            cv2.setMouseCallback("Body Consider", onMouse_RB2)

        elif right_body_page == 2:
            cv2.circle(img_Rbody, (887, 124), 5, (0, 0, 255), -1)
            cv2.imshow("Body Consider", img_Rbody)

            cv2.setMouseCallback("Body Consider", onMouse_RB2)

        elif right_body_page == 3:
            cv2.circle(img_Rbody, (887, 220), 5, (0, 0, 255), -1)
            cv2.imshow("Body Consider", img_Rbody)

            cv2.setMouseCallback("Body Consider", onMouse_RB2)

        elif right_body_page == 4:
            cv2.circle(img_Rbody, (887, 124), 5, (0, 0, 255), -1)
            cv2.circle(img_Rbody, (887, 220), 5, (0, 0, 255), -1)
            cv2.imshow("Body Consider", img_Rbody)

            cv2.setMouseCallback("Body Consider", onMouse_RB2)

def onMouse_RB2(event, x, y, flags, param):
    global right_body_plus

    global right_body_page

    img_Rbody = cv2.imread("C:/Users/{}/Desktop/img_Rbody.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 50 < y < 140:
        if right_body_page == 1:
            cv2.circle(img_Rbody, (887, 124), 5, (0, 0, 255), -1)
            cv2.imshow("Body Consider", img_Rbody)
            right_body_plus = 1
            right_body_page = 2

        elif right_body_page == 2:
            cv2.imshow("Body Consider", img_Rbody)
            right_body_plus = 0
            right_body_page = 1

        elif right_body_page == 3:
            cv2.circle(img_Rbody, (887, 220), 5, (0, 0, 255), -1)
            cv2.circle(img_Rbody, (887, 124), 5, (0, 0, 255), -1)
            cv2.imshow("Body Consider", img_Rbody)
            right_body_plus = 2
            right_body_page = 4

        elif right_body_page == 4:
            cv2.circle(img_Rbody, (887, 220), 5, (0, 0, 255), -1)
            cv2.imshow("Body Consider", img_Rbody)
            right_body_plus = 1
            right_body_page = 3

    elif event == cv2.EVENT_LBUTTONDOWN and 140 < y < 267:
        if right_body_page == 1:
            cv2.circle(img_Rbody, (887, 220), 5, (0, 0, 255), -1)
            cv2.imshow("Body Consider", img_Rbody)
            right_body_plus = 1
            right_body_page = 3

        elif right_body_page == 2:
            cv2.circle(img_Rbody, (887, 220), 5, (0, 0, 255), -1)
            cv2.circle(img_Rbody, (887, 124), 5, (0, 0, 255), -1)
            cv2.imshow("Body Consider", img_Rbody)
            right_body_plus = 2
            right_body_page = 4

        elif right_body_page == 3:
            cv2.imshow("Body Consider", img_Rbody)
            right_body_plus = 0
            right_body_page = 1

        elif right_body_page == 4:
            cv2.circle(img_Rbody, (887, 124), 5, (0, 0, 255), -1)
            cv2.imshow("Body Consider", img_Rbody)
            right_body_plus = 1
            right_body_page = 2

#==========================================================================================
def onMouse_RLeg1(event, x, y, flags, param):
    global right_leg_point

    global right_leg_page

    img_RLeg = cv2.imread("C:/Users/{}/Desktop/img_Rleg.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 260 < x < 280 and 550 < y < 570:
        if right_leg_page == 1:
            cv2.imshow("Leg Consider", img_RLeg)
            right_leg_point = 1

            cv2.setMouseCallback("Leg Consider", onMouse_RLeg2)

        elif right_leg_page == 2:
            cv2.circle(img_RLeg, (886, 135), 5, (0, 0, 255), -1)
            cv2.imshow("Leg Consider", img_RLeg)

            cv2.setMouseCallback("Leg Consider", onMouse_RLeg2)

def onMouse_RLeg2(event, x, y, flags, param):
    global right_leg_point

    global right_leg_page

    img_RLeg = cv2.imread("C:/Users/{}/Desktop/img_Rleg.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 45 < y < 165:
        if right_leg_page == 1:
            cv2.circle(img_RLeg, (886, 135), 5, (0, 0, 255), -1)
            cv2.imshow("Leg Consider", img_RLeg)
            right_leg_point = 2
            right_leg_page = 2

        elif right_leg_page == 2:
            cv2.imshow("Leg Consider", img_RLeg)
            right_leg_point = 1
            right_leg_page = 1
#========================================================================================
def onMouse_RWeight1(event, x, y, flags, param):
    # right weight
    global right_weight_plus
    global right_weight_page

    global right_muscle_plus
    global right_muscle_page

    img_Rweight = cv2.imread("C:/Users/{}/Desktop/img_Rweight.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 400 < x < 450 and 650 < y < 700:
        if right_muscle_page == 1:
            right_muscle_plus = 0
            if right_weight_page == 1:
                cv2.circle(img_Rweight, (887, 220), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)
                right_weight_plus = 0

                cv2.setMouseCallback("weight", onMouse_RWeight2)


            elif right_weight_page == 2:
                cv2.circle(img_Rweight, (887, 315), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)
                right_weight_plus = 1

                cv2.setMouseCallback("weight", onMouse_RWeight2)


            elif right_weight_page == 3:
                cv2.circle(img_Rweight, (887, 410), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)
                right_weight_plus = 2

                cv2.setMouseCallback("weight", onMouse_RWeight2)


            elif right_weight_page == 4:
                cv2.circle(img_Rweight, (887, 535), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)
                right_weight_plus = 3

                cv2.setMouseCallback("weight", onMouse_RWeight2)

        elif right_muscle_page == 2:
            right_muscle_plus = 1

            if right_weight_page == 1:
                cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
                cv2.circle(img_Rweight, (887, 220), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)
                right_weight_plus = 0

                cv2.setMouseCallback("weight", onMouse_RWeight2)


            elif right_weight_page == 2:
                cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
                cv2.circle(img_Rweight, (887, 315), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)
                right_weight_plus = 1

                cv2.setMouseCallback("weight", onMouse_RWeight2)


            elif right_weight_page == 3:
                cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
                cv2.circle(img_Rweight, (887, 410), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)
                right_weight_plus = 2

                cv2.setMouseCallback("weight", onMouse_RWeight2)


            elif right_weight_page == 4:
                cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
                cv2.circle(img_Rweight, (887, 535), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)
                right_weight_plus = 3

                cv2.setMouseCallback("weight", onMouse_RWeight2)

def onMouse_RWeight2(event, x, y, flags, param):
    # right weight
    global right_weight_plus
    global right_weight_page

    global right_muscle_page
    global right_muscle_plus

    img_Rweight = cv2.imread("C:/Users/{}/Desktop/img_Rweight.png".format(os.getlogin()))

    if event == cv2.EVENT_LBUTTONDOWN and 115 < y < 135:
        if right_muscle_page == 1 :
            if right_weight_page == 1 :
                cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
                cv2.circle(img_Rweight, (887, 220), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)

                right_weight_page = 1
                right_weight_plus = 0

                right_muscle_page = 2
                right_muscle_plus = 1

            elif right_weight_page == 2 :
                cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
                cv2.circle(img_Rweight, (887, 315), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)

                right_weight_page = 2
                right_weight_plus = 1

                right_muscle_page = 2
                right_muscle_plus = 1

            elif right_weight_page == 3 :
                cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
                cv2.circle(img_Rweight, (887, 410), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)

                right_weight_page = 3
                right_weight_plus = 2

                right_muscle_page = 2
                right_muscle_plus = 1

            elif right_weight_page == 4 :
                cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
                cv2.circle(img_Rweight, (887, 535), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)

                right_weight_page = 4
                right_weight_plus = 3

                right_muscle_page = 2
                right_muscle_plus = 1

        elif right_muscle_page == 2:
            if right_weight_page == 1:
                cv2.circle(img_Rweight, (887, 220), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)

                right_weight_page = 1
                right_weight_plus = 0

                right_muscle_page = 1
                right_muscle_plus = 0

            elif right_weight_page == 2:
                cv2.circle(img_Rweight, (887, 315), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)

                right_weight_page = 2
                right_weight_plus = 1

                right_muscle_page = 1
                right_muscle_plus = 0

            elif right_weight_page == 3:
                cv2.circle(img_Rweight, (887, 410), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)

                right_weight_page = 3
                right_weight_plus = 2

                right_muscle_page = 1
                right_muscle_plus = 0

            elif right_weight_page == 4:
                cv2.circle(img_Rweight, (887, 535), 5, (0, 0, 255), -1)
                cv2.imshow("weight", img_Rweight)

                right_weight_page = 4
                right_weight_plus = 3

                right_muscle_page = 1
                right_muscle_plus = 0


    elif event == cv2.EVENT_LBUTTONDOWN and 210 < y < 230:
        if right_muscle_page == 1 :
            cv2.circle(img_Rweight, (887, 220), 5, (0, 0, 255), -1)
            cv2.imshow("weight", img_Rweight)

            right_weight_page = 1
            right_weight_plus = 0

        elif right_muscle_page == 2:
            cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
            cv2.circle(img_Rweight, (887, 220), 5, (0, 0, 255), -1)
            cv2.imshow("weight", img_Rweight)

            right_weight_page = 1
            right_weight_plus = 0

    elif event == cv2.EVENT_LBUTTONDOWN and 305 < y < 325:
        if right_muscle_page == 1:
            cv2.circle(img_Rweight, (887, 315), 5, (0, 0, 255), -1)
            cv2.imshow("weight", img_Rweight)

            right_weight_page = 2
            right_weight_plus = 1

        elif right_muscle_page == 2:
            cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
            cv2.circle(img_Rweight, (887, 315), 5, (0, 0, 255), -1)
            cv2.imshow("weight", img_Rweight)

            right_weight_page = 2
            right_weight_plus = 1

    elif event == cv2.EVENT_LBUTTONDOWN and 400 < y < 420:
        if right_muscle_page == 1:
            cv2.circle(img_Rweight, (887, 415), 5, (0, 0, 255), -1)
            cv2.imshow("weight", img_Rweight)

            right_weight_page = 3
            right_weight_plus = 2

        elif right_muscle_page == 2:
            cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
            cv2.circle(img_Rweight, (887, 415), 5, (0, 0, 255), -1)
            cv2.imshow("weight", img_Rweight)

            right_weight_page = 3
            right_weight_plus = 2

    elif event == cv2.EVENT_LBUTTONDOWN and 525 < y < 545:
        if right_muscle_page == 1:
            cv2.circle(img_Rweight, (887, 535), 5, (0, 0, 255), -1)
            cv2.imshow("weight", img_Rweight)

            right_weight_page = 4
            right_weight_plus = 3

        elif right_muscle_page == 2:
            cv2.circle(img_Rweight, (887, 125), 5, (0, 0, 255), -1)
            cv2.circle(img_Rweight, (887, 535), 5, (0, 0, 255), -1)
            cv2.imshow("weight", img_Rweight)

            right_weight_page = 4
            right_weight_plus = 3

#=========================================================================================

def call_func():
    if

    # cv2.setMouseCallback("BodyPoint", onMouse_RU1)
    # cv2.waitKey(10)
    # cv2.setMouseCallback("BodyPoint", onMouse_RL1)
    # cv2.waitKey(10)
    # cv2.setMouseCallback("BodyPoint", onMouse_RN1)
    # cv2.waitKey(10)
    # cv2.setMouseCallback("BodyPoint", onMouse_RW_and_RWT1)
    # cv2.waitKey(100)
    # cv2.setMouseCallback("BodyPoint", onMouse_RB1)
    # cv2.waitKey(100)
    # cv2.setMouseCallback("BodyPoint", onMouse_RLeg1)
    # cv2.waitKey(100)
    # cv2.setMouseCallback("BodyPoint", onMouse_RWeight1)
    # cv2.waitKey(100)

#=======================================================================================================================
def loadWebCam_left():
    global left_lower_point
    global left_upper_point
    global left_wrist_point
    global left_neck_point
    global left_body_point
    global left_leg_point

    global left_Score_A, left_Score_B, left_Score_Grand

    # left visualization
    groupA_point_left = open('C:/Users/{}/Desktop/My Program/Left/Point/groupA_point_left.csv'.format(os.getlogin()), 'w', newline='')
    groupB_point_left = open('C:/Users/{}/Desktop/My Program/Left/Point/groupB_point_left.csv'.format(os.getlogin()), 'w', newline='')
    grand_point_left = open('C:/Users/{}/Desktop/My Program/Left/Point/grand_point_left.csv'.format(os.getlogin()), 'w', newline='')

    groupA_point_left_csv = csv.writer(groupA_point_left)
    R_groupA_point_left = ['left_upper_point', 'left_lower_point', 'left_wrist_point']
    groupA_point_left_csv.writerow(R_groupA_point_left)

    groupB_point_left_csv = csv.writer(groupB_point_left)
    R_groupB_point_left = ['left_body_point', 'left_neck_point']
    groupB_point_left_csv.writerow(R_groupB_point_left)

    grand_point_left_csv = csv.writer(grand_point_left)
    R_grand_point_left = ['left_grand_point']
    grand_point_left_csv.writerow(R_grand_point_left)

    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    camera_video = cv2.VideoCapture(1)
    camera_video.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    camera_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start = time.time() # 시작 시간 저장

    start_now = datetime.now()
    print("시작시각은 : ", start_now.strftime('%Y-%m-%d %H:%M:%S'), "입니다.")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 영상을 기록할 코덱 설정
    is_record = False  # 녹화상태는 처음엔 거짓으로 설정
    is_capture = False

    # Initialize a resizable window.
    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pose Classification', 1080, 720)

    while camera_video.isOpened():

        now = datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        nowDatetime_path = now.strftime('%Y-%m-%d %H_%M_%S')

        ok, frame = camera_video.read()

        if not ok:
            continue

        cv2.putText(frame, text=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

        frame_height, frame_width, _ = frame.shape

        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        frame, landmarks = detectPose(frame, pose_video)

        if landmarks:
            frame = classifyPose_left(landmarks, frame)

        ################################################ 추가내용 ############################################
        R_groupA_point_left = [left_upper_point, left_lower_point, left_wrist_point]
        groupA_point_left_csv.writerow(R_groupA_point_left)

        R_groupB_point_left = [left_body_point, left_neck_point]
        groupB_point_left_csv.writerow(R_groupB_point_left)

        R_grand_point_left = [left_Score_Grand]
        grand_point_left_csv.writerow(R_grand_point_left)
        #######################################################################################################

        key = cv2.waitKey(1) & 0xFF

        ################################ For Visualization 가즈아ㅏㅏㅏ#############################
        '''
        img = cv2.imread("bodypoint_L.jpg") # 이미지 불러오기
        resize_img = cv2.resize(img, (480, 720), interpolation=cv2.INTER_CUBIC) # 사이즈 재조정~
        '''

        img = cv2.imdecode(L_image_nparray, cv2.IMREAD_COLOR)
        resize_img = cv2.resize(img, (480, 720), interpolation=cv2.INTER_CUBIC)  # 사이즈 재조정~v

        global left_upper_count, left_lower_count, left_wrist_count, left_body_count, left_neck_count

        if left_upper_point == 3 or left_upper_point == 4:
            left_upper_count += 1

        if left_upper_count <= 280:
            cv2.circle(resize_img, (162, 250), int(1 + (left_upper_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif left_upper_count > 280:
            cv2.circle(resize_img, (162, 250), 15, (0, 0, 255), -1) # 노가다 원(빨간원)


        if left_lower_point == 3:
            left_lower_count += 1

        if left_lower_count <= 280:
            cv2.circle(resize_img, (140, 330), int(1 + (left_lower_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif left_lower_count > 280:
            cv2.circle(resize_img, (140, 330), 15, (0, 0, 255), -1) # 노가다 원(빨간원)


        if left_wrist_point == 3 or left_wrist_point == 4:
            left_wrist_count += 1

        if left_wrist_count <= 280:
            cv2.circle(resize_img, (126, 370), int(1 + (left_wrist_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif left_wrist_count > 280:
            cv2.circle(resize_img, (126, 370), 15, (0, 0, 255), -1) # 노가다 원(빨간원)


        if left_neck_point == 3 or left_neck_point == 4:
            left_neck_count += 1

        if left_neck_count <= 280:
            cv2.circle(resize_img, (240, 150), int(1 + (left_neck_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif left_neck_count > 280:
            cv2.circle(resize_img, (240, 150), 15, (0, 0, 255), -1) # 노가다 원(빨간원)


        if left_body_point == 3 or left_body_point == 4:
            left_body_count += 1

        if left_body_count <= 280:
            cv2.circle(resize_img, (240, 220), int(1 + (left_body_count*0.05)), (0, 0, 255), -1) # 노가다 원(빨간원)
        elif left_body_count > 280:
            cv2.circle(resize_img, (240, 220), 15, (0, 0, 255), -1) # 노가다 원(빨간원)

            # legs
        if left_leg_point == 1:
            cv2.circle(resize_img, (270, 560), 3, (0, 255, 0), -1) # 노가다 원(초록원)

        elif left_leg_point == 2:
            cv2.circle(resize_img, (270, 560), 15, (0, 0, 255), -1) # 노가다 원(빨간원)

        cv2.imshow("BodyPoint", resize_img) # 출력


        if key == ord('r') and is_record == False:
            is_record = True
            video = cv2.VideoWriter("C:/Users/{}/Desktop/My Program/Left/Record/".format(os.getlogin()) + nowDatetime_path + ".mp4", fourcc, 10,
                                        (frame.shape[1], frame.shape[0]))
        elif key == ord('r') and is_record == True:
            is_record = False
            video.release()

        elif key == ord('c'):
            is_capture = True
            start_time = time.time()
            capture_ext = ".png"

            im1 = frame
            im2 = resize_img

            def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
                h_min = min(im.shape[0] for im in im_list)
                im_list_resize = [
                    cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                    for im in im_list]
                return cv2.hconcat(im_list_resize)

            im_h_resize = hconcat_resize_min([im1, im2])
            cv2.imwrite("C:/Users/{}/Desktop/My Program/Left/Captured/{}{}".format(os.getlogin(), nowDatetime_path, capture_ext), im_h_resize)

        elif key == ord('q'):
            break

        elif (key == 27):
            groupA_point_left.close()
            groupB_point_left.close()
            grand_point_left.close()
            folder_path = "C:/Users/{}/Desktop".format(os.getlogin())
            folder_path = os.path.realpath(folder_path)
            os.startfile(folder_path)
            break


        if is_record == True:
            video.write(frame)
            cv2.circle(img=frame, center=(620, 15), radius=5, color=(0, 0, 255), thickness=-1)

        if is_capture == True:
            cv2.putText(frame, "Captured", org=(900, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(0, 255, 0), thickness=2)
            if time.time() - start_time > 2:
                is_capture = False

        # 왼쪽 자동캡쳐 코드
        global img_counter_left_upper
        global img_counter_left_lower
        global img_counter_left_wrist
        global img_counter_left_neck
        global img_counter_left_body
        global img_counter_left_grand_score
        global left_upper_capture_time, left_upper_capture_time2
        global left_lower_capture_time, left_lower_capture_time2
        global left_wrist_capture_time, left_wrist_capture_time2
        global left_body_capture_time, left_body_capture_time2
        global left_neck_capture_time, left_neck_capture_time2
        global left_grand_capture_time, left_grand_capture_time2

        if left_upper_point > 3:

            left_upper_capture_time = time.time()

            if abs(left_upper_capture_time2 - left_upper_capture_time) > 1:
                img_name_left_upper = "left_upper{}.png".format(img_counter_left_upper)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/left_upper/'.format(os.getlogin()) + img_name_left_upper,
                            frame)
                print("{} written!".format(img_name_left_upper))
                img_counter_left_upper += 1
                left_upper_capture_time2 = time.time()

        if left_lower_point > 1:

            left_lower_capture_time = time.time()

            if abs(left_lower_capture_time2 - left_lower_capture_time) > 1:
                img_name_left_lower = "left_lower{}.png".format(img_counter_left_lower)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/left_lower/'.format(os.getlogin()) + img_name_left_lower,
                            frame)
                print("{} written!".format(img_name_left_lower))
                img_counter_left_lower += 1
                left_lower_capture_time2 = time.time()

        if left_wrist_point > 2:

            left_wrist_capture_time = time.time()

            if abs(left_wrist_capture_time2 - left_wrist_capture_time) > 1:
                img_name_left_wrist = "left_wrist{}.png".format(img_counter_left_wrist)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/left_wrist/'.format(os.getlogin()) + img_name_left_wrist,
                            frame)
                print("{} written!".format(img_name_left_wrist))
                img_counter_left_wrist += 1
                left_wrist_capture_time2 = time.time()

        if left_neck_point > 2:

            left_neck_capture_time = time.time()

            if abs(left_neck_capture_time2 - left_neck_capture_time) > 1:
                img_name_left_neck = "left_neck{}.png".format(img_counter_left_neck)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/left_neck/'.format(os.getlogin()) + img_name_left_neck,
                            frame)
                print("{} written!".format(img_name_left_neck))
                img_counter_left_neck += 1
                left_neck_capture_time2 = time.time()

        if left_body_point > 2:

            left_body_capture_time = time.time()

            if abs(left_body_capture_time2 - left_body_capture_time) > 1:
                img_name_left_body = "left_body{}.png".format(img_counter_left_body)
                cv2.imwrite('C:/Users/{}/Desktop/My Program/Right/Captured/left_body/'.format(os.getlogin()) + img_name_left_body,
                            frame)
                print("{} written!".format(img_name_left_body))
                img_counter_left_body += 1
                left_body_capture_time2 = time.time()

        if left_Score_Grand > 2:

            left_grand_capture_time = time.time()

            if abs(left_grand_capture_time2 - left_grand_capture_time) > 1:
                img_name_left_grand_score = "left_grand_score{}.png".format(img_counter_left_grand_score)
                cv2.imwrite(
                    'C:/Users/{}/Desktop/My Program/Right/Captured/left_grand_score/'.format(os.getlogin()) + img_name_left_grand_score,
                    frame)
                print("{} written!".format(img_name_left_grand_score))
                img_counter_left_grand_score += 1
                left_grand_capture_time2 = time.time()

        running_time = math.trunc(time.time() - start)

        cv2.putText(frame, "Time : " + str(running_time), (230, 70), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 255, 0), 2)

        cv2.imshow("Pose Classification", frame)

    camera_video.release()
    cv2.destroyAllWindows()

    ################################ 추가내용 (프로그램 작동시간) #############################
    work_time = int(time.time() - start)
    print("time:", work_time) # 현재시각 - 시작시간 = 실행시간

    end_now = datetime.now()
    print("끝나는 시각은 : ", end_now.strftime('%Y-%m-%d %H:%M:%S'), "입니다.")


    #################################### 추가내용 Data analysis ###############################
    # 데이터프레임 가져오기
    data_groupA_point_left = pd.read_csv('C:/Users/{}/Desktop/My Program/Left/Point/groupA_point_left.csv'.format(os.getlogin()))
    data_groupB_point_left = pd.read_csv('C:/Users/{}/Desktop/My Program/Left/Point/groupB_point_left.csv'.format(os.getlogin()))
    data_grand_point_left = pd.read_csv('C:/Users/{}/Desktop/My Program/Left/Point/grand_point_left.csv'.format(os.getlogin()))


    # 데이터의 갯수
    data_groupA_numbers_left = data_groupA_point_left['left_upper_point'].count()
    print(data_groupA_numbers_left)
    data_groupB_numbers_left = data_groupB_point_left['left_body_point'].count()
    print(data_groupB_numbers_left)
    print(data_grand_point_left['left_grand_point'].count())
    # groupA랑 groupB의 데이터 갯수는 똑같음


    # 시간을 포함한 데이터프레임 완성
    number_per_sec_left = round(data_groupA_numbers_left/work_time, 1)  # 초당 데이터 갯수

    sec = []
    for x in range(work_time):
        while len(sec) != int(number_per_sec_left*(x + 1)):
                sec.append(x)
    print(sec)
    len(sec)

    sec = pd.Series(sec, name='Time')

    data_groupA_point_left = pd.concat((data_groupA_point_left,sec), axis=1)
    data_groupB_point_left = pd.concat((data_groupB_point_left,sec), axis=1)
    data_grand_point_left = pd.concat((data_grand_point_left,sec), axis=1)

    data_groupA_point_left = data_groupA_point_left.dropna() # 반올림으로 인해 1개까지 차이가 날 수 있음. 그걸 제거
    data_groupB_point_left = data_groupB_point_left.dropna() # 1개 제거하더라도 대세엔 지장이 없음
    data_grand_point_left = data_grand_point_left.dropna()

    data_groupA_point_left = data_groupA_point_left.groupby(by='Time').mean()
    data_groupB_point_left = data_groupB_point_left.groupby(by='Time').mean()
    data_grand_point_left = data_grand_point_left.groupby(by='Time').mean()


    # Visualization
    # 그룹A 그래프
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plt.ylabel('Left groupA Point')
    sns.lineplot(data=data_groupA_point_left, x="Time", y="left_upper_point", label="left_upper")
    sns.lineplot(data=data_groupA_point_left, x="Time", y="left_lower_point", label="left_lower")
    sns.lineplot(data=data_groupA_point_left, x="Time", y="left_wrist_point", label="left_wrist")
    plt.savefig('C:/Users/{}/Desktop/My Program/Left/Graph/GroupA_Graph_left.png'.format(os.getlogin()))


    # 그룹B 그래프
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plt.ylabel('Left groupB Point')
    sns.lineplot(data=data_groupB_point_left, x="Time", y="left_body_point", label="left_body")
    sns.lineplot(data=data_groupB_point_left, x="Time", y="left_neck_point", label="left_neck")
    plt.savefig('C:/Users/{}/Desktop/My Program/Left/Graph/GroupB_Graph_left.png'.format(os.getlogin()))


    # 최종점수 그래프
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    plt.ylabel('Left Grand Score')
    sns.lineplot(data=data_grand_point_left, x="Time", y="left_grand_point", label="left_grand_point")
    plt.savefig('C:/Users/{}/Desktop/My Program/Left/Graph/Grand_Score_Graph_left.png'.format(os.getlogin()))
    

    # 데이터 요약
    print("관절별 부하의 정도")
    mean_of_left_upper_point = data_groupA_point_left["left_upper_point"].mean()
    mean_of_left_lower_point = data_groupA_point_left["left_lower_point"].mean()
    mean_of_left_wrist_point = data_groupA_point_left["left_wrist_point"].mean()
    mean_of_left_body_point = data_groupB_point_left["left_body_point"].mean()
    mean_of_left_neck_point = data_groupB_point_left["left_neck_point"].mean()
    print("left_upper_point의 평균부하의 정도 :", mean_of_left_upper_point)
    print("left_lower_point의 평균부하의 정도 :", mean_of_left_lower_point)
    print("left_wrist_point의 평균부하의 정도 :", mean_of_left_wrist_point)
    print("left_body_point의 평균부하의 정도 :", mean_of_left_body_point)
    print("left_neck_point의 평균부하의 정도 :", mean_of_left_neck_point)


#=======================================================================================================================

def LR():
    root = tk.Toplevel(main)
    root.title("측정 기준")
    root.geometry("450x325+100+100")
    root.resizable(False, False)

    label = tk.Label(root, text="동작분석 보조프로그램", font='나눔바른펜')
    label.place(y=25, relx=0.33)

    cam_btn_R = tk.Button(root, text='Right', font='Consolas', command=loadWebCam_right)
    cam_btn_R.place(x=150, y=100, width=150, height=50)

    cam_btn_L = tk.Button(root, text='Left', font='Consolas', command=loadWebCam_left)
    cam_btn_L.place(x=150, y=200, width=150, height=50)

    # 배치 좌표
    center_x = int((screenwidth - 450) / 2)
    center_y = int((screenheight - 325) / 2)

    # 크기와 좌표 지정
    root.geometry("{}x{}+{}+{}".format(450, 325, center_x, center_y))
    root.resizable(False, False)

    root.mainloop()



# 초기창 실행
main = tk.Tk()
main.title("자동화된 동작분석에 기반한 근골격계 작업부하 평가 지원")

# 실행창 이름 부분
main.overrideredirect(False)

# # url image test
url = "https://i.postimg.cc/bw3G6kJY/image.png"
urllib.request.urlretrieve(url, "main.png")

# 이미지 크기
# img = ImageTk.PhotoImage(Image.open("C:/Users/{}/Desktop/전체 이미지.png".format(os.getlogin())))
img = ImageTk.PhotoImage(Image.open("main.png"))

height = img.height()
width = img.width()

# screen 크기
screenwidth = main.winfo_screenwidth()
screenheight = main.winfo_screenheight()

# 배치 좌표
center_x = int((screenwidth - width)/2)
center_y = int((screenheight - height)/2)

# 크기와 좌표 지정
main.geometry("{}x{}+{}+{}".format(width, height, center_x, center_y))
main.resizable(False, False)

# 메인화면 배치 하기
label = Label(image=img)
label.pack()

# 버튼 Font
btn_font = font.Font(size=34, weight="bold")

# START 버튼
btn_start = Button(main, text="START", command=LR)
btn_start.config(width=6, height=1) # 버튼 가로 세로 크기 변경
btn_start.place(relx=0.77, rely=0.8)
btn_start["font"] = btn_font


main.mainloop()

