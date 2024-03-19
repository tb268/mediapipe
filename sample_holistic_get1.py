#!/usr/bin/env python
# -*- coding: utf-8 -*-
#python3 sample_holistic_get2.py --plot_world_landmark
#xが左右、yが奥行き、zが上下
#行動取得用

import copy
import argparse

import cv2
import numpy as np
import mediapipe as mp
import csv
from utils import CvFpsCalc
from sqltest import sql_set
import time


from tqdm import tqdm

#角度計算(3座標から)
def cal_slope(a,b,c):
    # 点A,B,Cの座標（3次元座標上の場合）
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # ベクトルを定義
    vec_a = a - b
    vec_c = c - b

    # コサインの計算
    length_vec_a = np.linalg.norm(vec_a)
    length_vec_c = np.linalg.norm(vec_c)
    inner_product = np.inner(vec_a, vec_c)
    cos = inner_product / (length_vec_a * length_vec_c)

    # 角度（ラジアン）の計算
    rad = np.arccos(cos)

    # 弧度法から度数法（rad ➔ 度）への変換
    degree = np.rad2deg(rad)

    return degree


#角度計算(2ベクトルから)
def cal_slope_for_vector(a_0,a_1,c_0,c_1):
    a_0 = np.array(a_0)
    a_1 = np.array(a_1)
    c_0 = np.array(c_0)
    c_1 = np.array(c_1)


    # ベクトルを定義
    vec_a = a_0 - a_1
    vec_c = c_0 - c_1

    # コサインの計算
    length_vec_a = np.linalg.norm(vec_a)
    length_vec_c = np.linalg.norm(vec_c)
    inner_product = np.inner(vec_a, vec_c)
    cos = inner_product / (length_vec_a * length_vec_c)

    # 角度（ラジアン）の計算
    rad = np.arccos(cos)

    # 弧度法から度数法（rad ➔ 度）への変換
    degree = np.rad2deg(rad)

    return degree

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # parser.add_argument('--upper_body_only', action='store_true')  # 0.8.3 or less
    parser.add_argument('--unuse_smooth_landmarks', action='store_true')
    parser.add_argument('--enable_segmentation', action='store_true')
    parser.add_argument('--smooth_segmentation', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--segmentation_score_th",
                        help='segmentation_score_threshold',
                        type=float,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')
    parser.add_argument('--plot_world_landmark', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    #DBのテーブルをクリア

    """
    dbname = 'main.db'
    with sqlite3.connect(dbname) as conn:
        # SQLiteを操作するためのカーソルを作成
        cur = conn.cursor()
        cur.execute("DELETE FROM pose")
        # コミットしないと登録が反映されない
        conn.commit()
    
    """
    
    #csvをクリア
    with open("posedata.csv", "w") as f:
        pass
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # upper_body_only = args.upper_body_only
    smooth_landmarks = not args.unuse_smooth_landmarks
    enable_segmentation = args.enable_segmentation
    smooth_segmentation = args.smooth_segmentation
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    segmentation_score_th = args.segmentation_score_th

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark

    # カメラ準備 ###############################################################
    #cap = cv2.VideoCapture(cap_device)
    cap=cv2.VideoCapture('harehare.mp4')
    #cap=cv2.VideoCapture('testm.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        smooth_segmentation=smooth_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # World座標プロット ########################################################
    if plot_world_landmark:
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    
    isFirst=True #最初のフレームかどうか（BDテーブル削除用）

    with tqdm() as pbar: #進捗状況取得用#############################
        while True:
            pbar.update(1)
            display_fps = cvFpsCalc.get()

            # カメラキャプチャ #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv2.flip(image, 1)  # ミラー表示
            debug_image = copy.deepcopy(image)

            # 検出実施 #############################################################
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Face Mesh ###########################################################
            """
            face_landmarks = results.face_landmarks
            if face_landmarks is not None:
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, face_landmarks)
                # 描画
                debug_image = draw_face_landmarks(debug_image, face_landmarks)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                """
            # Pose （描画用のみ）###############################################################
            if enable_segmentation and results.segmentation_mask is not None:
                # セグメンテーション
                mask = np.stack((results.segmentation_mask, ) * 3,
                                axis=-1) > segmentation_score_th
                bg_resize_image = np.zeros(image.shape, dtype=np.uint8)
                bg_resize_image[:] = (0, 255, 0)
                debug_image = np.where(mask, debug_image, bg_resize_image)

            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, pose_landmarks)
                # 描画
                debug_image = draw_pose_landmarks(
                    debug_image,
                    pose_landmarks,
                    # upper_body_only,
                )
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            
            # Pose:World座標プロット（3D座標） #############################################
            if plot_world_landmark:
                if results.pose_world_landmarks is not None:
                    landmark_point=plot_world_landmarks(
                        plt,
                        ax,
                        results.pose_world_landmarks,
                    )

            landmark_point = []#有効性、xyz座標

            for index, landmark in enumerate(results.pose_world_landmarks.landmark):
                landmark_point.append(
                    [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

            #csv書き込み＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
            
            #データ出力
            with open("./posedata.csv","a") as f:
                writer=csv.writer(f)
                writer.writerow(landmark_point)#有効性、xyz座標
            now=time.time()
            #sql書き込み。最初のフレームならDBクリア
            sql_set(now,landmark_point,"pose1",isFirst)
            
            isFirst=False
            # Hands ###############################################################
            """
            left_hand_landmarks = results.left_hand_landmarks
            right_hand_landmarks = results.right_hand_landmarks
            # 左手
            if left_hand_landmarks is not None:
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, left_hand_landmarks)
                # 描画
                debug_image = draw_hands_landmarks(
                    debug_image,
                    cx,
                    cy,
                    left_hand_landmarks,
                    # upper_body_only,
                    'R',
                )
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            # 右手
            if right_hand_landmarks is not None:
                # 手の平重心計算
                cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, right_hand_landmarks)
                # 描画
                debug_image = draw_hands_landmarks(
                    debug_image,
                    cx,
                    cy,
                    right_hand_landmarks,
                    # upper_body_only,
                    'L',
                )
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            """

            # FPS表示
            
            if enable_segmentation and results.segmentation_mask is not None:
                fps_color = (255, 255, 255)
            else:
                fps_color = (0, 255, 0)
            cv2.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, fps_color, 2, cv2.LINE_AA)
        
            # キー処理(ESC：終了) #################################################
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

            # 画面反映 #############################################################
            #cv2.imshow('MediaPipe Holistic Demo', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv2.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_hands_landmarks(
        image,
        cx,
        cy,
        landmarks,
        # upper_body_only,
        handedness_str='R'):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

        # if not upper_body_only:
        if True:
            cv2.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv2.LINE_AA)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv2.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv2.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # 人差指
        cv2.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv2.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv2.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # 中指
        cv2.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv2.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv2.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # 薬指
        cv2.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv2.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv2.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # 小指
        cv2.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv2.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv2.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # 手の平
        cv2.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv2.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv2.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv2.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv2.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv2.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv2.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        cv2.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv2.putText(image, handedness_str, (cx - 6, cy + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return image


def draw_face_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        cv2.circle(image, (landmark_x, landmark_y), 1, (0, 255, 0), 1)

    if len(landmark_point) > 0:
        # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

        # 左眉毛(55：内側、46：外側)
        cv2.line(image, landmark_point[55], landmark_point[65], (0, 255, 0), 2)
        cv2.line(image, landmark_point[65], landmark_point[52], (0, 255, 0), 2)
        cv2.line(image, landmark_point[52], landmark_point[53], (0, 255, 0), 2)
        cv2.line(image, landmark_point[53], landmark_point[46], (0, 255, 0), 2)

        # 右眉毛(285：内側、276：外側)
        cv2.line(image, landmark_point[285], landmark_point[295], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[295], landmark_point[282], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[282], landmark_point[283], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[283], landmark_point[276], (0, 255, 0),
                2)

        # 左目 (133：目頭、246：目尻)
        cv2.line(image, landmark_point[133], landmark_point[173], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[173], landmark_point[157], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[157], landmark_point[158], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[158], landmark_point[159], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[159], landmark_point[160], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[160], landmark_point[161], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[161], landmark_point[246], (0, 255, 0),
                2)

        cv2.line(image, landmark_point[246], landmark_point[163], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[163], landmark_point[144], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[144], landmark_point[145], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[145], landmark_point[153], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[153], landmark_point[154], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[154], landmark_point[155], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[155], landmark_point[133], (0, 255, 0),
                2)

        # 右目 (362：目頭、466：目尻)
        cv2.line(image, landmark_point[362], landmark_point[398], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[398], landmark_point[384], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[384], landmark_point[385], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[385], landmark_point[386], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[386], landmark_point[387], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[387], landmark_point[388], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[388], landmark_point[466], (0, 255, 0),
                2)

        cv2.line(image, landmark_point[466], landmark_point[390], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[390], landmark_point[373], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[373], landmark_point[374], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[374], landmark_point[380], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[380], landmark_point[381], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[381], landmark_point[382], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[382], landmark_point[362], (0, 255, 0),
                2)

        # 口 (308：右端、78：左端)
        cv2.line(image, landmark_point[308], landmark_point[415], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[415], landmark_point[310], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[310], landmark_point[311], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[311], landmark_point[312], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[312], landmark_point[13], (0, 255, 0), 2)
        cv2.line(image, landmark_point[13], landmark_point[82], (0, 255, 0), 2)
        cv2.line(image, landmark_point[82], landmark_point[81], (0, 255, 0), 2)
        cv2.line(image, landmark_point[81], landmark_point[80], (0, 255, 0), 2)
        cv2.line(image, landmark_point[80], landmark_point[191], (0, 255, 0), 2)
        cv2.line(image, landmark_point[191], landmark_point[78], (0, 255, 0), 2)

        cv2.line(image, landmark_point[78], landmark_point[95], (0, 255, 0), 2)
        cv2.line(image, landmark_point[95], landmark_point[88], (0, 255, 0), 2)
        cv2.line(image, landmark_point[88], landmark_point[178], (0, 255, 0), 2)
        cv2.line(image, landmark_point[178], landmark_point[87], (0, 255, 0), 2)
        cv2.line(image, landmark_point[87], landmark_point[14], (0, 255, 0), 2)
        cv2.line(image, landmark_point[14], landmark_point[317], (0, 255, 0), 2)
        cv2.line(image, landmark_point[317], landmark_point[402], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[402], landmark_point[318], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[318], landmark_point[324], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[324], landmark_point[308], (0, 255, 0),
                2)

    return image

#ポーズのワールド座標取得===================================================================
def draw_pose_landmarks(
    image,
    landmarks,
    # upper_body_only,
    visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # 鼻
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 右目：目頭
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 右目：瞳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 右目：目尻
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 左目：目頭
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 左目：瞳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 左目：目尻
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 右耳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 左耳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 口：左端
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 口：左端
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 右肩
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  # 右肘
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 右手首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 左手首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  # 右手1(外側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 左手1(外側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 右手2(先端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 左手2(先端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 右手3(内側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 左手3(内側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 腰(右側)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:  # 腰(左側)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  # 右ひざ
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  # 左ひざ
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:  # 右足首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  # 左足首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  # 右かかと
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 左かかと
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 右つま先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 左つま先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        # if not upper_body_only:
        if True:
            cv2.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv2.LINE_AA)

    if len(landmark_point) > 0:
        # 右目
        if landmark_point[1][0] > visibility_th and landmark_point[2][
                0] > visibility_th:
            cv2.line(image, landmark_point[1][1], landmark_point[2][1],
                    (0, 255, 0), 2)
        if landmark_point[2][0] > visibility_th and landmark_point[3][
                0] > visibility_th:
            cv2.line(image, landmark_point[2][1], landmark_point[3][1],
                    (0, 255, 0), 2)

        # 左目
        if landmark_point[4][0] > visibility_th and landmark_point[5][
                0] > visibility_th:
            cv2.line(image, landmark_point[4][1], landmark_point[5][1],
                    (0, 255, 0), 2)
        if landmark_point[5][0] > visibility_th and landmark_point[6][
                0] > visibility_th:
            cv2.line(image, landmark_point[5][1], landmark_point[6][1],
                    (0, 255, 0), 2)

        # 口
        if landmark_point[9][0] > visibility_th and landmark_point[10][
                0] > visibility_th:
            cv2.line(image, landmark_point[9][1], landmark_point[10][1],
                    (0, 255, 0), 2)

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[12][1],
                    (0, 255, 0), 2)

        # 右腕
        if landmark_point[11][0] > visibility_th and landmark_point[13][
                0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[13][1],
                    (0, 255, 0), 2)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv2.line(image, landmark_point[13][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][
                0] > visibility_th:
            cv2.line(image, landmark_point[12][1], landmark_point[14][1],
                    (0, 255, 0), 2)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv2.line(image, landmark_point[14][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][
                0] > visibility_th:
            cv2.line(image, landmark_point[15][1], landmark_point[17][1],
                    (0, 255, 0), 2)
        if landmark_point[17][0] > visibility_th and landmark_point[19][
                0] > visibility_th:
            cv2.line(image, landmark_point[17][1], landmark_point[19][1],
                    (0, 255, 0), 2)
        if landmark_point[19][0] > visibility_th and landmark_point[21][
                0] > visibility_th:
            cv2.line(image, landmark_point[19][1], landmark_point[21][1],
                    (0, 255, 0), 2)
        if landmark_point[21][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv2.line(image, landmark_point[21][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][
                0] > visibility_th:
            cv2.line(image, landmark_point[16][1], landmark_point[18][1],
                    (0, 255, 0), 2)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
                0] > visibility_th:
            cv2.line(image, landmark_point[18][1], landmark_point[20][1],
                    (0, 255, 0), 2)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
                0] > visibility_th:
            cv2.line(image, landmark_point[20][1], landmark_point[22][1],
                    (0, 255, 0), 2)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv2.line(image, landmark_point[22][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 胴体
        if landmark_point[11][0] > visibility_th and landmark_point[23][
                0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[23][1],
                    (0, 255, 0), 2)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv2.line(image, landmark_point[12][1], landmark_point[24][1],
                    (0, 255, 0), 2)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv2.line(image, landmark_point[23][1], landmark_point[24][1],
                    (0, 255, 0), 2)

        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                    0] > visibility_th:
                cv2.line(image, landmark_point[23][1], landmark_point[25][1],
                        (0, 255, 0), 2)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                    0] > visibility_th:
                cv2.line(image, landmark_point[25][1], landmark_point[27][1],
                        (0, 255, 0), 2)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                    0] > visibility_th:
                cv2.line(image, landmark_point[27][1], landmark_point[29][1],
                        (0, 255, 0), 2)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                    0] > visibility_th:
                cv2.line(image, landmark_point[29][1], landmark_point[31][1],
                        (0, 255, 0), 2)

            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                    0] > visibility_th:
                cv2.line(image, landmark_point[24][1], landmark_point[26][1],
                        (0, 255, 0), 2)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                    0] > visibility_th:
                cv2.line(image, landmark_point[26][1], landmark_point[28][1],
                        (0, 255, 0), 2)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                    0] > visibility_th:
                cv2.line(image, landmark_point[28][1], landmark_point[30][1],
                        (0, 255, 0), 2)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                    0] > visibility_th:
                cv2.line(image, landmark_point[30][1], landmark_point[32][1],
                        (0, 255, 0), 2)
    
    return image

#ポーズのワールド座標を取得
def plot_world_landmarks(
    plt,
    ax,
    landmarks,
    visibility_th=0.5,
):
    landmark_point = []#有効性、xyz座標

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append(
            [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]#肩、肘、手首、外側端、先端、内側端
    left_arm_index_list = [12, 14, 16, 18, 20, 22]#肩、肘、手首、外側端、先端、内側端
    right_body_side_index_list = [12, 24, 26, 28, 30, 32]#肩、腰、膝、足首、かかと、足先
    left_body_side_index_list = [11, 23, 25, 27, 29, 31]#肩、腰、膝、足首、かかと、足先
    shoulder_index_list = [11, 12]#左肩、右肩
    waist_index_list = [23, 24]#左腰、右腰


    # 顔
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    
    for index in right_arm_index_list:
        #print(point)
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))
    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))
    
    #csv書き込み＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    """
    #データ出力
    with open("./posedata.csv","a") as f:
        writer=csv.writer(f)
        writer.writerow(landmark_point)#有効性、xyz座標
        #print(' '.join(landmark_point))
        #writer.writerow(' '.join(landmark_point))#有効性、xyz座標
    """
    
    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)
    plt.xlabel('X-label')
    plt.ylabel('Y-label')
    plt.pause(.001)

    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


if __name__ == '__main__':    #開始時間
    start=time.time()#開始時間
    main()
    now=time.time()#終了時間
    times=now-start
    print("経過時間"+str(times)+"秒")
