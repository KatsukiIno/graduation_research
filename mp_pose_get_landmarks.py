# mediapipe pose 「m」キーを押すとcsv出力開始

import cv2
import copy
import itertools
import mediapipe as mp
import pandas as pd
import csv
import os
import numpy as np
import tensorflow as tf
from collections import Counter

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

input_video_path = 'input/videos/pose_videos/pointing_center_right.mp4'
output_csv_path = 'model/pose_classifier/test_24_11_07.csv'
pose_label = 15
pose_target_indices = {0} | set(range(11, 23))
hand_target_indices = [0, 4, 8, 12, 16, 20]

interpreter = tf.lite.Interpreter(model_path="model/pose_classifier/pose_classifier_11_25.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def main():
  cap = cv2.VideoCapture(0) #webcam:「0」

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_count = 1
  mode = False

  pose_history = []

  
  with mp_pose.Pose(
    model_complexity = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as pose:
    
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # video:「break」, webcam:「continue」
        break

      # 後で自分撮りビューを表示するために画像を水平方向に反転し、BGR画像をRGBに変換
      image = cv2.flip(image, 1)  # ミラー表示
      debug_image = copy.deepcopy(image)
      
      # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      results = pose.process(image)
      image.flags.writeable = True

      display_image = np.ones((600, 300, 3), dtype=np.uint8) * 255  # 白背景の画像
    
      if results.pose_landmarks is not None:

        brect = calc_bounding_rect(debug_image, results.pose_landmarks)
        pose_landmarks = calc_landmark_list(debug_image, results.pose_landmarks, pose_target_indices)
        pre_processed_pose_landmark_list = pre_process_landmark(pose_landmarks)
        draw_specific_pose_landmarks(debug_image, results.pose_landmarks, pose_target_indices, (0, 0, 255))
        debug_image = draw_bounding_rect(debug_image, brect)
        debug_image = draw_info_text(debug_image, brect, None)

        pose_probs = None
        # TFLiteモデルに入力
        input_data = np.array([pre_processed_pose_landmark_list], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pose_probs = output_data[0]  # 各クラスの確率が含まれる

        debug_image = draw_pose_info(debug_image, np.argmax(pose_probs), pose_history)

        if mode:
          logging_csv(pre_processed_pose_landmark_list)
        else:
          display_image = draw_class_probabilities(display_image, pose_probs)

        image.flags.writeable = True

      frame_count += 1
      
      cv2.putText(debug_image, f"frame_count:{frame_count} / {total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.putText(debug_image, f"Mode: {'ON' if mode else 'OFF'}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      
      cv2.imshow('MediaPipe Pose', debug_image)
      cv2.imshow('Class Probabilities', display_image)
      key = cv2.waitKey(10) & 0xFF  # 一度だけ呼び出す
      if key == 27:
        break
      elif key == 109:  # 'm' キーが押されたときに mode を切り替え
        mode = not mode
        print(f"Mode changed to: {'ON' if mode else 'OFF'}")
        
  cap.release()
  cv2.destroyAllWindows()
  

def draw_pose_info(image, pose_prob, pose_history):
    # ラベルを取得
    
    # ラベルが no_pose の場合、履歴をクリア
    if pose_prob == 14:
        pose_history.clear()
        return image

    # ラベルが履歴にない場合、履歴に追加
    if not pose_history or pose_history[-1] != get_label(pose_prob):
      if len(pose_history) < 2:
        pose_history.append(get_label(pose_prob))

    # 表示するテキストを作成
    display_text = " --> ".join(pose_history)

    # 背景の描画
    h, w, _ = image.shape
    overlay_height = 50  # 背景の高さ
    overlay_color = (50, 50, 50)  # 灰色
    text_color = (255, 255, 255)  # 白色

    # 背景矩形の描画
    image = cv2.rectangle(image, (0, h - overlay_height), (w, h), overlay_color, -1)

    # テキストの描画
    font_scale = 0.8
    thickness = 2
    y_offset = h - overlay_height + 30
    cv2.putText(image, display_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    return image


# クラスインデックスを受け取りラベルを返す関数
def get_label(class_index):
  label_dict = {
    0:"left_pointing",
    1:"right_pointing",
    2:"left_t_pose",
    3:"right_t_pose",
    4:"left_arm_folding",
    5:"right_arm_folding",
    6:"left_hand_raising",
    7:"right_hand_raising",
    8:"raise_hands_to_chest",
    9:"left_catched_ball",
    10:"right_catched_ball",
    11:"end_of_set",
    12:"substitutions",
    13:"pointing_center_left",
    14:"other(no_pose)",
    15:"pointing_center_right"
    }
  return label_dict.get(class_index, "Unknown")  # 該当がない場合 "Unknown" を返す  

def draw_class_probabilities(image, pose_probs):
  y_offset = 50
  if pose_probs is not None:
    cv2.putText(image, "pose", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for i, prob in enumerate(pose_probs):
      y_offset += 25
      cv2.putText(image, f"{i}: {prob:.1%}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    y_offset += 50
    pose_label = get_label(np.argmax(pose_probs))
    cv2.putText(image, pose_label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
  else:
    cv2.putText(image, "None", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  
  return image

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

def draw_specific_pose_landmarks(image, landmarks, indices, color):
  for idx in indices:
    if landmarks.landmark[idx].visibility > 0.5:  # You can adjust the visibility threshold
      landmark = landmarks.landmark[idx]
      h, w, c = image.shape
      cx, cy = int(landmark.x * w), int(landmark.y * h)
      cv2.circle(image, (cx, cy), 5, color, cv2.FILLED)

def draw_specific_hand_landmarks(image, landmarks, indices, color):
  for idx in indices:
    landmark = landmarks.landmark[idx]
    h, w, c = image.shape
    cx, cy = int(landmark.x * w), int(landmark.y * h)
    cv2.circle(image, (cx, cy), 5, color, cv2.FILLED)


def logging_csv(landmark_list):
  # CSVファイルにデータを追記する
  with open(output_csv_path, 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow([pose_label, *landmark_list])  # データ行を追記する


def calc_landmark_list(image, landmarks, target_indices):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # キーポイント
    for idx, landmark in enumerate(landmarks.landmark):
        
        if idx not in target_indices:
            continue  # 指定されたインデックス範囲外のランドマークは無視
        
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

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
    temp_landmark_list = landmark_list.copy()

    base_x, base_y = temp_landmark_list[0], temp_landmark_list[1]

    for i in range(0, len(temp_landmark_list), 2):
      temp_landmark_list[i] -= base_x
      temp_landmark_list[i+1] -= base_y

    max_value = max(abs(x) for x in temp_landmark_list if x != "NaN")

    def normalize_(n):
        return n / max_value if n != "NaN" else "NaN"

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list



# def pre_process_pose_landmark(landmark_list):
#     temp_landmark_list = copy.deepcopy(landmark_list)

#     # 相対座標に変換
#     base_x, base_y = 0, 0
#     for index, landmark_point in enumerate(temp_landmark_list):
#         if index == 0:
#             base_x, base_y = landmark_point[0], landmark_point[1]

#         temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
#         temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

#     # 1次元リストに変換
#     temp_landmark_list = list(
#         itertools.chain.from_iterable(temp_landmark_list))

#     # 正規化
#     max_value = max(list(map(abs, temp_landmark_list)))

#     def normalize_(n):
#         return n / max_value

if __name__ == '__main__':
  main()