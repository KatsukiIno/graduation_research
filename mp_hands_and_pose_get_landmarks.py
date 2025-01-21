# mediapipe hands 全てのキーポイントをcsvデータに出力

import cv2
import copy
import itertools
import mediapipe as mp
import pandas as pd
import csv
import os
import numpy as np
import tensorflow as tf
from collections import Counter, deque

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# input_video_path = "input/videos/hand_videos/one.mp4"
input_video_path = "input/videos/action_videos/8_both_hands_raising/16.mp4"
output_csv_path = 'model/hand_classifier/mp_hands_test.csv'
pose_label = 0
pose_target_indices = {0} | set(range(11, 23))
hand_target_indices = [0, 4, 8, 12, 16, 20]
# mode = False

queue_length = 10
class_queue = deque(maxlen=queue_length)

interpreter_hand = tf.lite.Interpreter(model_path="model/hand_classifier/hands_classifier_11_22.tflite")
interpreter_hand.allocate_tensors()

interpreter_pose = tf.lite.Interpreter(model_path="model/pose_classifier/pose_classifier_12_15.tflite")
interpreter_pose.allocate_tensors()


def main():
  cap = cv2.VideoCapture(input_video_path) #webcam:「0」
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_count = 1
  mode = False
  pose_probs = None
  pose_history = []
  class_label = None
  prov_class_label = None
  
  with mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose, \
  mp_hands.Hands(max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5) as hands:

    
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # video:「break」, webcam:「continue」
        break

      image = cv2.flip(image, 1)  # ミラー表示 内カメラでの撮影であれば必要
      debug_image = copy.deepcopy(image)
      
      # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      pose_results = pose.process(image)
      hands_results = hands.process(image)
      image.flags.writeable = True

      hands_display_image = np.ones((400, 300, 3), dtype=np.uint8) * 255  # 白背景の画像
      pose_display_image = np.ones((600, 300, 3), dtype=np.uint8) * 255  # 白背景の画像

      if np.argmax(pose_probs) in [6, 7, 8]:

        if hands_results.multi_hand_landmarks is not None:
          for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):

            brect = calc_bounding_rect(debug_image, hand_landmarks)
            
            hand_landmark_list = calc_hand_landmark_list(debug_image, hand_landmarks, hand_target_indices)
            pre_processed_hand_landmark = pre_process_landmark(hand_landmark_list)
            draw_specific_hand_landmarks(debug_image, hands_results.multi_hand_landmarks, hand_target_indices, (0, 0, 255), handedness)
            debug_image = draw_bounding_rect(debug_image, brect)
            debug_image = draw_info_text(debug_image, brect, handedness)

            hand_input_data = np.array([pre_processed_hand_landmark], dtype=np.float32)
            hand_probs = run_model(interpreter_hand, hand_input_data)
            right_probs = None
            left_probs = None

            if handedness.classification[0].label == "Right":
              right_probs = hand_probs
            else:
              left_probs = hand_probs

            if mode:
              logging_csv(pre_processed_hand_landmark, handedness)
            else:
              hands_display_image = draw_hands_class_probabilities(hands_display_image, right_probs, left_probs)


      if pose_results.pose_landmarks is not None:

        brect = calc_bounding_rect(debug_image, pose_results.pose_landmarks)
        pose_landmarks = calc_landmark_list(debug_image, pose_results.pose_landmarks, pose_target_indices)
        pre_processed_pose_landmark = pre_process_landmark(pose_landmarks)
        draw_specific_pose_landmarks(debug_image, pose_results.pose_landmarks, pose_target_indices, (0, 255, 0))
        debug_image = draw_bounding_rect(debug_image, brect)

        pose_input_data = np.array([pre_processed_pose_landmark], dtype=np.float32)
        pose_probs = run_model(interpreter_pose, pose_input_data)

        # pose_history.append(get_pose_label(np.argmax(pose_probs)))

        debug_image = draw_info_text(debug_image, brect, None)

        # ///////////////////////////////////////////////////////
        class_label, class_queue = update_class_queue(np.argmax(pose_probs), 7)
        if prov_class_label != class_label:
          print(f"最頻クラス：{get_pose_label(class_label)}")
          print(f"クラスキュー：{class_queue}")
          print()
        prov_class_label = class_label
        # ///////////////////////////////////////////////////////

        if mode:
          logging_csv(pre_processed_pose_landmark)
        else:
          pose_display_image = draw_pose_class_probabilities(pose_display_image, pose_probs)
      else:
        pose_probs = None

      frame_count += 1
      
      cv2.putText(debug_image, f"frame_count:{frame_count} / {total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.putText(debug_image, f"Mode: {'ON' if mode else 'OFF'}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if mode else (0, 0, 255), 2, cv2.LINE_AA)
      cv2.rectangle(debug_image, (40, 310), (300, 360), (128, 128, 128), -1)
      cv2.putText(debug_image, f"{get_pose_label(class_label)}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
      
      cv2.imshow('MediaPipe Hands', debug_image)
      cv2.imshow('Class pose Probabilities', pose_display_image)
      cv2.imshow('Class hands Probabilities', hands_display_image)

      key = cv2.waitKey(10) & 0xFF  # 一度だけ呼び出す
      if key == 27:
        break
      elif key == 109:  # 'm' キーが押されたときに mode を切り替え
        mode = not mode
        print(f"Mode changed to: {'ON' if mode else 'OFF'}")
        
  cap.release()
  cv2.destroyAllWindows()




def update_class_queue(new_class, M):
    """
    キューに新しいクラスを追加し、最頻クラスを条件に応じて返す。
    
    Args:
        new_class (int or str): 新しいクラス
        M (int): 最頻クラスと判定するための最低出現数

    Returns:
        tuple: 最頻クラス (条件を満たさない場合は None) と現在のキュー
    """
    # 新しいクラスをキューに追加
    class_queue.append(new_class)
    
    # キュー内のクラス頻度を計算
    class_counts = Counter(class_queue)
    
    # 最頻クラスとその出現数を取得
    most_common_class, count = class_counts.most_common(1)[0]
    
    # M個以上出現していればそのクラスを返す
    if count >= M:
        return most_common_class, list(class_queue)
    else:
        return None, list(class_queue)


def get_pose_label(class_index):
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
    9:"end_of_set",
    10:"substitutions",
    11:"other(no_pose)"
    }
  return label_dict.get(class_index, "Unknown")  # 該当がない場合 "Unknown" を返す  


def get_hands_label(class_index):
  label_dict = {
    0:"scissors",
    1:"four",
    2:"three",
    3:"good",
    4:"other",
    5:"one"
    }
  return label_dict.get(class_index, "Unknown")  # 該当がない場合 "Unknown" を返す

def draw_pose_class_probabilities(image, pose_probs):
  y_offset = 50
  if pose_probs is not None:
    cv2.putText(image, "pose", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for i, prob in enumerate(pose_probs):
      y_offset += 25
      cv2.putText(image, f"{i}: {prob:.1%}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    y_offset += 50
    pose_label = get_pose_label(np.argmax(pose_probs))
    cv2.putText(image, pose_label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
  else:
    cv2.putText(image, "None", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  
  return image

def draw_hands_class_probabilities(image, right_probs, left_probs):
  w = image.shape[1]
  y_offset = 50
  if left_probs is not None:
    cv2.putText(image, "left_hands", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for i, prob in enumerate(left_probs):
      y_offset += 25
      cv2.putText(image, f"{i}: {prob:.1%}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    y_offset += 50
    left_hand_label = get_hands_label(np.argmax(left_probs))
    cv2.putText(image, left_hand_label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
  else:
    y_offset += 25
    cv2.putText(image, None, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)


  # 右手の確率を描画
  y_offset = 50
  if right_probs is not None:
    cv2.putText(image, "right_hands", (w // 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for i, prob in enumerate(right_probs):
      y_offset += 25
      cv2.putText(image, f"{i}: {prob:.1%}", (w // 2 , y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    y_offset += 50
    right_hand_label = get_hands_label(np.argmax(right_probs))
    cv2.putText(image, right_hand_label, (w // 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
  else:
    y_offset += 25
    cv2.putText(image, None, (w // 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
  
  return image

# モデルの実行を行う関数
def run_model(interpreter, input_data):
  
  interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
  interpreter.invoke()
  output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
  
  return output_data[0]

def draw_info_text(image, brect, handedness):
  h, w, _ = image.shape

  cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
  
  if handedness is not None:
    info_text = handedness.classification[0].label[0:]
  else:
    info_text = "pose"
  cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

  return image


# def draw_info_text_sample(image, brect, handedness, pose_history):

  h, w, _ = image.shape

  # ハンド情報を描画
  cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
  if handedness is not None:
    info_text = handedness.classification[0].label[0:]
  else:
    info_text = "pose"
  cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

  # ポーズ履歴を描画
  if pose_history:
    # 矩形描画 (画面下部)
    rect_x1, rect_y1 = 50, int(h - 100)
    rect_x2, rect_y2 = int(w - 50), int(h - 50)
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (75, 75, 75), -1)

    # 履歴テキスト作成
    display_text = " -> ".join(pose_history)

    # テキストサイズを取得して中央揃え位置を計算
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
    text_width, text_height = text_size[0], text_size[1]

    text_x = rect_x1 + (rect_x2 - rect_x1 - text_width) // 2
    text_y = rect_y1 + (rect_y2 - rect_y1 + text_height) // 2

    # テキスト描画
    cv2.putText(image, display_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

  return image

def draw_bounding_rect(image, brect):
  # 外接矩形
  cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                (0, 0, 0), 1)

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

def draw_specific_hand_landmarks(image, landmarks, indices, color, handedness):
  # for idx in indices:
  #   landmark = landmarks.landmark[idx]
  #   h, w, c = image.shape
  #   cx, cy = int(landmark.x * w), int(landmark.y * h)
  #   cv2.circle(image, (cx, cy), 5, color, cv2.FILLED)
  # if handedness.classification[0].label == "Right":
  #   color = (255, 0, 0)
  # else:
  #   color = (0, 0, 255)

  if landmarks is not None:
    for hand_landmarks in landmarks:  # 各手のランドマークを個別に処理
        for idx in indices:
            landmark = hand_landmarks.landmark[idx]
            # 必要な描画処理を実行
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, color, -1)


def logging_csv(landmark_list, handedness):
  # CSVファイルにデータを追記する
  with open(output_csv_path, 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow([pose_label, handedness.classification[0].label[0:], *landmark_list])  # データ行を追記する


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


def calc_hand_landmark_list(image, landmarks, target_indices):
  image_width, image_height = image.shape[1], image.shape[0]
  
  landmark_point = []

  for idx in target_indices:  # target_indicesを使って特定のランドマークを取得
    landmark = landmarks.landmark[idx]
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