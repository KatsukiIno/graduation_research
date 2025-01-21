# mediapipe hands ハンドサイン推論 + 「m」キーで座標をcsv出力

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
mp_hands = mp.solutions.hands

input_video_path = "videos/left_hand_piece_test.mp4"
output_csv_path = 'model/hand_classifier/hands_classifier_11_27.csv'
output_images_path = r'D:\\images\\hands\\good'
pose_label = 3
pose_target_indices = {0} | set(range(13, 23))
hand_target_indices = [0, 4, 8, 12, 16, 20]

interpreter = tf.lite.Interpreter(model_path="model/hand_classifier/sample.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def main():
  cap = cv2.VideoCapture(0) #webcam:「0」
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_count = 1
  save_count = 213
  mode = False

  
  with mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:
    
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # video:「break」, webcam:「continue」
        break

      image = cv2.flip(image, 1)  # ミラー表示
      debug_image = copy.deepcopy(image)
      
      # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      results = hands.process(image)
      image.flags.writeable = True

      display_image = np.ones((400, 300, 3), dtype=np.uint8) * 255  # 白背景の画像

      if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

          brect = calc_bounding_rect(debug_image, hand_landmarks)
          
          hand_landmark_list = calc_hand_landmark_list(debug_image, hand_landmarks, hand_target_indices)
          pre_process_hand_landmark = pre_process_landmark(hand_landmark_list)
          draw_specific_hand_landmarks(debug_image, results.multi_hand_landmarks, hand_target_indices, (0, 0, 255))
          debug_image = draw_bounding_rect(debug_image, brect)
          debug_image = draw_info_text(debug_image, brect, handedness)


          # TFLiteモデルに入力
          input_data = np.array([pre_process_hand_landmark], dtype=np.float32)
          interpreter.set_tensor(input_details[0]['index'], input_data)
          interpreter.invoke()
          output_data = interpreter.get_tensor(output_details[0]['index'])
          probabilities = output_data[0]  # 各クラスの確率が含まれる
          
          right_probs = None
          left_probs = None
          if handedness.classification[0].label == "Right":
            right_probs = probabilities
          else:
            left_probs = probabilities

          if mode:
            save_count += 1
            logging_csv(pre_process_hand_landmark, handedness)
            output_path = os.path.join(output_images_path, f"{save_count}.jpg")
            cv2.imwrite(output_path, debug_image[brect[1]-20:brect[3]+20, brect[0]-20:brect[2]+20])
            print(f"{save_count}.png：保存しました")
            mode = not mode
          else:
            display_image = draw_class_probabilities(display_image, right_probs, left_probs)

      frame_count += 1
      
      cv2.putText(debug_image, f"frame_count:{frame_count} / {total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
      # cv2.putText(debug_image, f"Mode: {'ON' if mode else 'OFF'}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(debug_image, f"save_count: {save_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      
      cv2.imshow('MediaPipe Hands', debug_image)
      cv2.imshow('Class Probabilities', display_image)
      key = cv2.waitKey(10) & 0xFF  # 一度だけ呼び出す
      if key == 27:
        break
      elif key == 109:  # 'm' キーが押されたときに mode を切り替え
        mode = not mode
        # print(f"Mode changed to: {'ON' if mode else 'OFF'}")
        
  cap.release()
  cv2.destroyAllWindows()


# クラスインデックスを受け取りラベルを返す関数
def get_label(class_index):
  label_dict = {
    0:"scissors",
    1:"four",
    2:"three",
    3:"good",
    4:"other",
    5:"one"
    }
  return label_dict.get(class_index, "Unknown")  # 該当がない場合 "Unknown" を返す


def draw_class_probabilities(image, right_probs, left_probs):
  w = image.shape[1]
  y_offset = 50
  if left_probs is not None:
    cv2.putText(image, "left_hands", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for i, prob in enumerate(left_probs):
      y_offset += 25
      cv2.putText(image, f"{i}: {prob:.1%}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    y_offset += 50
    left_hand_label = get_label(np.argmax(left_probs))
    cv2.putText(image, left_hand_label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

  # 右手の確率を描画
  y_offset = 50
  if right_probs is not None:
    cv2.putText(image, "right_hands", (w // 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for i, prob in enumerate(right_probs):
      y_offset += 25
      cv2.putText(image, f"{i}: {prob:.1%}", (w // 2 , y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    y_offset += 50
    right_hand_label = get_label(np.argmax(right_probs))
    cv2.putText(image, right_hand_label, (w // 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
  
  return image

def draw_info_text(image, brect, handedness):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

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

def draw_specific_hand_landmarks(image, landmarks, indices, color):
  # for idx in indices:
  #   landmark = landmarks.landmark[idx]
  #   h, w, c = image.shape
  #   cx, cy = int(landmark.x * w), int(landmark.y * h)
  #   cv2.circle(image, (cx, cy), 5, color, cv2.FILLED)

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
    writer.writerow([pose_label, *landmark_list])  # データ行を追記する


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