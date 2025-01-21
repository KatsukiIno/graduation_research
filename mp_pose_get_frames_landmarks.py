# mediapipe pose 「m」キーを押すと過去25フレーム分をcsv出力 

import cv2
import copy
import itertools
import mediapipe as mp
import pandas as pd
import numpy as np
import csv
import os
import tensorflow as tf
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# input_video_path = r"D:\\videos\\test_1.mp4"
input_video_path = "input/videos/action_videos/no_pose_test.mp4"
# input_video_path = "input/videos/action_videos/left_ball_touch_100.mp4"
output_csv_path = 'model/action_classifier/test_data_12_10.csv'
pose_label = 12

pose_target_indices = {0} | set(range(11, 23))
hand_target_indices = [0, 4, 8, 12, 16, 20]

history_length = 30
past_positions_15 = deque(maxlen=history_length)
past_positions_16 = deque(maxlen=history_length)
point_history = deque(maxlen=history_length)

interpreter_pose = tf.lite.Interpreter(model_path="model/pose_classifier/pose_classifier_12_15.tflite")
interpreter_pose.allocate_tensors()

def main():
    cap = cv2.VideoCapture(input_video_path)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    mode = False

    with mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        frame_count = 0
        save_count = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            

            if results.pose_landmarks is not None:
                brect = calc_bounding_rect(debug_image, results.pose_landmarks)
                pose_landmarks = calc_landmark_list(debug_image, results.pose_landmarks, pose_target_indices)
                pre_processed_pose_landmark = pre_process_landmark(pose_landmarks)
                draw_specific_pose_landmarks(debug_image, results.pose_landmarks, pose_target_indices, (0, 0, 255))
                debug_image = draw_bounding_rect(debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, None)
            
                pose_input_data = np.array([pre_processed_pose_landmark], dtype=np.float32)
                pose_probs = run_model(interpreter_pose, pose_input_data)

                point_history.append(pre_processed_pose_landmark)
                past_positions_15.append(results.pose_landmarks.landmark[15])
                past_positions_16.append(results.pose_landmarks.landmark[16])
                debug_image = draw_past_positions(debug_image, past_positions_15, past_positions_16)
                frame_count += 1

                # key = cv2.waitKey(1) & 0xFF

                # バッファが溜まったら保存
            if mode and len(point_history) == history_length:
                logging_csv(point_history, pose_label)  # CSVにデータを保存
                frame_count = 0  # フレームカウントをリセット
                mode = False
                    # save_count += 1

                    
                    # 数字キーの処理 (1～9)
                    # if ord('0') <= key <= ord('9'):
                    #     label = chr(key)
                    # # 代用キーの処理 ('a'～'f' を 10～15に対応)
                    # elif ord('a') <= key <= ord('f'):
                    #     label = str(key - ord('a') + 10)  # 'a' -> 10, 'b' -> 11, ..., 'f' -> 15
                    # else:
                    #     label = None

                    # if label and len(point_history) == history_length:
                    #     logging_csv(point_history, label)  # CSVにデータを保存
                    #     frame_count = 0  # フレームカウントをリセット
                    #     key_save_count[label] += 1  # ラベルの保存数を更新

            # display_save_count(debug_image, key_save_count, frame_count, mode)
            cv2.putText(debug_image, 'frame_count : ' + str(frame_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, 'save_count : ' + str(save_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('MediaPipe Holistic', debug_image)
            key = cv2.waitKey(5)
            if key & 0xFF == 27:  # ESCキーで終了
                break
            elif key & 0xFF == ord('m'):  # 'm'キーでモード切替
                mode = True
                save_count += 1

    cap.release()
    cv2.destroyAllWindows()

# ///////////////////////////////////////////////////


def run_model(interpreter, input_data):
  
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    return output_data[0]


def display_save_count(debug_image, key_save_count, frame_count, mode):
    y_offset = 60  # 表示開始位置（Y座標）
    cv2.putText(debug_image, 'frame_count : ' + str(frame_count), (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    y_offset += 30
    cv2.putText(debug_image, 'mode : ' + str(mode), (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    y_offset += 30
    for label, count in key_save_count.items():
        text = f'Label {label}: {count}'
        cv2.putText(debug_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        y_offset += 30  # 次のラベルは下に表示


def draw_past_positions(frame, past_positions_15, past_positions_16):
    """ランドマークをピクセル座標に変換して過去の位置を描画する関数"""
    h, w, _ = frame.shape
    for i, (lm_15, lm_16) in enumerate(zip(past_positions_15, past_positions_16)):
        # ピクセル座標に変換
        pos_15 = (int(lm_15.x * w), int(lm_15.y * h))
        pos_16 = (int(lm_16.x * w), int(lm_16.y * h))
        
        # 最新が小さく、過去が大きい円の半径
        radius = int(10 * (i / len(past_positions_15))) + 1
        color = (0, 255, 0)  # 緑色の円

        # 円を塗りつぶしなしで描画
        cv2.circle(frame, pos_15, radius, color, 1)
        cv2.circle(frame, pos_16, radius, color, 1)
    return frame

def draw_info_text(image, brect, handedness):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    if handedness is not None:
        info_text = handedness.classification[0].label[0:]
    else:
        info_text = "pose"
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def draw_bounding_rect(image, brect):
    # 外接矩形
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_specific_pose_landmarks(image, landmarks, indices, color):
  for idx in indices:
    if landmarks.landmark[idx].visibility > 0.5:  # You can adjust the visibility threshold
        landmark = landmarks.landmark[idx]
        h, w, c = image.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, color, cv2.FILLED)


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


def draw_bounding_box_hand(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    x_coords = []
    y_coords = []

    for landmark in landmarks.landmark:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        x_coords.append(x)
        y_coords.append(y)

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

def logging_csv(point_history, label):
    with open(output_csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        for landmarks_row in point_history:
            writer.writerow([label, *landmarks_row])

def calc_landmark_list(image, landmarks, target_indices):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # キーポイント
    for idx, landmark in enumerate(landmarks.landmark):
        if idx not in target_indices:
            continue  # 指定されたインデックス範囲外のランドマークは無視

        # if landmark.visibility < MIN_VISIBILITY:
        #     landmark_x = "NaN"
        #     landmark_y = "NaN"
        # else:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

        landmark_point.extend([landmark_x, landmark_y])

    return landmark_point

def calc_hand_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.extend([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0], temp_landmark_list[1]

    for i in range(0, len(temp_landmark_list), 2):
        temp_landmark_list[i] -= base_x
        temp_landmark_list[i+1] -= base_y

    max_value = max(abs(x) for x in temp_landmark_list if x != "NaN")

    def normalize_(n):
        return n / max_value if n != "NaN" else "NaN"

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

if __name__ == '__main__':
    main()
