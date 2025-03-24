# MediaPipe holistic
# 上半身骨格座標：MLP 手骨格座標：MLP

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
mp_holistic = mp.solutions.holistic

input_video_path = "input/videos/action_videos/3.mp4"
# input_video_path = "input/videos/referee_action_videos/delay_in_service.mp4"
# auto_score_video.mp4"
# output_csv_path = 'model/hand_classifier/mp_hands_test.csv'
pointing_csv_path = 'output/csv/score_record/MLP_scoring_sheet_1.csv'
pose_label = 0
pose_target_indices = {0} | set(range(11, 23))
hand_target_indices = [0, 4, 8, 12, 16, 20]

queue_length = 10
pose_queue = deque(maxlen=queue_length)
right_queue = deque(maxlen=queue_length)
left_queue = deque(maxlen=queue_length)

interpreter_hand = tf.lite.Interpreter(model_path="model/hand_classifier/hands_classifier_11_22.tflite")
interpreter_hand.allocate_tensors()

interpreter_pose = tf.lite.Interpreter(model_path="model/pose_classifier/pose_classifier_12_15.tflite")
interpreter_pose.allocate_tensors()


def main():
  cap = cv2.VideoCapture(input_video_path) #webcam:「0」
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_count = 1
  pose_probs = None
  class_label = None
  prov_class_label = None
  prov_left_hand_class_label = None
  prov_right_hand_class_label = None
  pose_history = []
  left_hand_history = []
  right_hand_history = []
  most_common_class_left_hand = None 
  most_common_class_right_hand = None
  pose_threshold = 7
  
  pose_class_txt = None

  
  left_team_point = 0
  right_team_point = 0
  serve_side = None

  count = 1

  with mp_holistic.Holistic(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as holisitc:

    
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # video:「break」, webcam:「continue」
        break

      image = cv2.flip(image, 1)  # ミラー表示 内カメラでの撮影であれば必要
      debug_image = copy.deepcopy(image)
      
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      results = holisitc.process(debug_image)
      image.flags.writeable = True

      pose_class_display = np.ones((600, 600, 3), dtype=np.uint8) * 255  # 白背景の画像



      if pose_history:

        if pose_history[-1] in [6, 8]:
          if results.right_hand_landmarks is not None:
            brect = calc_bounding_rect(debug_image, results.right_hand_landmarks)
            
            hand_landmark_list = calc_hand_landmark_list(debug_image, results.right_hand_landmarks, hand_target_indices)
            pre_processed_hand_landmark = pre_process_landmark(hand_landmark_list)
            draw_specific_hand_landmarks(debug_image, results.right_hand_landmarks, hand_target_indices, (0, 0, 255), None)
            debug_image = draw_bounding_rect(debug_image, brect)
            debug_image = draw_info_text(debug_image, brect, None)

            hand_input_data = np.array([pre_processed_hand_landmark], dtype=np.float32)
            hand_probs = run_model(interpreter_hand, hand_input_data)
            right_probs = hand_probs

            # ////////////////////////////////////////////////
            right_hand_history.append(np.argmax(right_probs))
            # ////////////////////////////////////////////////

        else:
          if right_hand_history:
            most_common_class_right_hand = Counter(right_hand_history).most_common(1)[0][0]
            # print(f"右手最多クラス：{Counter(right_hand_history).most_common(1)[0][0]}，{get_hands_label(Counter(right_hand_history).most_common(1)[0][0])}")
            # print(right_hand_history)
            right_hand_history.clear()

        if pose_history[-1] in [7, 8]:
          if results.left_hand_landmarks is not None:
            brect = calc_bounding_rect(debug_image, results.left_hand_landmarks)
            
            hand_landmark_list = calc_hand_landmark_list(debug_image, results.left_hand_landmarks, hand_target_indices)
            pre_processed_hand_landmark = pre_process_landmark(hand_landmark_list)
            draw_specific_hand_landmarks(debug_image, results.left_hand_landmarks, hand_target_indices, (0, 0, 255), None)
            debug_image = draw_bounding_rect(debug_image, brect)
            debug_image = draw_info_text(debug_image, brect, None)

            hand_input_data = np.array([pre_processed_hand_landmark], dtype=np.float32)
            hand_probs = run_model(interpreter_hand, hand_input_data)
            left_probs = hand_probs

            left_hand_history.append(np.argmax(left_probs))  
        else:
          if left_hand_history:
            most_common_class_left_hand = Counter(left_hand_history).most_common(1)[0][0]
            # print(f"左手最多クラス：{Counter(left_hand_history).most_common(1)[0][0]}，{get_hands_label(Counter(left_hand_history).most_common(1)[0][0])}")
            # print(left_hand_history)
            left_hand_history.clear()

        if most_common_class_left_hand is not None or most_common_class_right_hand is not None:
          # print(f"right_hand:{most_common_class_right_hand}, left_hand:{most_common_class_left_hand}")
          hands_class = get_hands_class(most_common_class_left_hand, most_common_class_right_hand)
          print(f"{hands_class}: [{most_common_class_left_hand, most_common_class_right_hand}]")
          pose_class_txt = hands_class
          most_common_class_left_hand = None 
          most_common_class_right_hand = None



      if results.pose_landmarks is not None:

        brect = calc_bounding_rect(debug_image, results.pose_landmarks)
        pose_landmarks = calc_landmark_list(debug_image, results.pose_landmarks, pose_target_indices)
        pre_processed_pose_landmark = pre_process_landmark(pose_landmarks)
        draw_specific_pose_landmarks(debug_image, results.pose_landmarks, pose_target_indices, (0, 255, 0))
        debug_image = draw_bounding_rect(debug_image, brect)

        pose_input_data = np.array([pre_processed_pose_landmark], dtype=np.float32)
        pose_probs = run_model(interpreter_pose, pose_input_data)

        debug_image = draw_info_text(debug_image, brect, None)
  

        # ///////////////////////////////////////////////////////
        class_label, pose_history_queue = update_class_queue(0, np.argmax(pose_probs), pose_threshold)
        if class_label is not None:
          if prov_class_label != class_label:
            if prov_class_label == 11 and len(pose_history) > 2:
              pose_history.clear()
              print(f"{count} / 10")
              print("pose_history cleared")
              print("")
              count += 1

            pose_history.append(class_label)
            print(f"pose_history:{pose_history}")
            # print(f"pose_queue{pose_history_queue}")
            prov_class_label = class_label
            pose_history, pose_class = get_combination_label(pose_history)
            if pose_class is not None:
              print(pose_class)
              pose_class_txt = pose_class

        

        # ///////////////////////////////////////////////////////

        pose_class_display = draw_pose_class(pose_class_display, pose_history)
      else:
        pose_probs = None

      # left_team_point, right_team_point, serve_side = auto_pointing(pose_history, left_team_point, right_team_point, serve_side)
      
      frame_count += 1
      
      debug_image = print_pose_class(debug_image, pose_class_txt, frame_count, total_frames)
      
      cv2.imshow('MediaPipe Hands', debug_image)
      cv2.imshow('pose class', pose_class_display)

      key = cv2.waitKey(10) & 0xFF  # 一度だけ呼び出す
      if key == 27:
        break
        
  cap.release()
  cv2.destroyAllWindows()



# ////////////////////////////////////////////////////////////////////

def print_pose_class(image, pose_class, frame_count, total_frames):
  
  h, w = image.shape[:2]
  font_scale = 1
  thickness = 2
  font = cv2.FONT_HERSHEY_SIMPLEX
  text_size, _ = cv2.getTextSize(pose_class, font, font_scale, thickness)
  text_width, text_height = text_size

  text_x = (w - text_width) // 2
  text_y = h - 20

  rect_start = (text_x - 10, text_y - text_height - 10)
  rect_end = (text_x + text_width + 10, text_y + 10)

  cv2.putText(image, f"frame_count:{frame_count} / {total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
  cv2.rectangle(image, rect_start, rect_end, (128, 128, 128), -1)
  if pose_class is not None:
    cv2.putText(image, pose_class, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

  return image

def get_hands_class(left_hand_label, right_hand_label):
    # 定義された組み合わせ辞書
    hand_combination_dict = {
        (0, None): "double contact",  # 右手だけ0の場合
        (None, 0): "double contact",  # 左手だけ0の場合
        (1, None): "four hit",
        (None, 1): "four hit",
        (2, 4): "delay in service",
        (4, 2): "delay in service",
        (3, 3): "no count",
        (4, 4): "ball out"
        # 他の組み合わせをここに追加
    }

    # 引数をタプルとして取得
    hands_tuple = (left_hand_label, right_hand_label)

    # 組み合わせ辞書から該当する文字列を取得
    if hands_tuple in hand_combination_dict:
      return hand_combination_dict[hands_tuple]
    
    # 該当なしの場合のデフォルト文字列
    return "no sign"


def create_score_board(left_point, right_point, serve_side):
  board = np.ones((400, 600, 3), dtype=np.uint8) * 255  # 白背景の画像
  h, w = board.shape[:2]
  padding = int(h * 0.1)  # 上下の余白
  team_y = padding         # チーム名のY座標
  score_y = int(h / 2)     # スコアのY座標
  serve_y = int(h * 0.8)   # サーブ表示のY座標
  left_x = int(w * 0.25)   # 左チームのX座標
  right_x = int(w * 0.75)  # 右チームのX座標

  # テキストのサイズを調整
  font_scale = 1.0
  font_thickness = 2

  cv2.putText(board, "Left Team", (left_x - int(w * 0.1), team_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
  cv2.putText(board, "Right Team", (right_x - int(w * 0.1), team_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
  cv2.putText(board, str(left_point), (left_x - int(w * 0.05), score_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
  cv2.putText(board, str(right_point), (right_x - int(w * 0.05), score_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)

  # サーブの表示（赤い円）
  if serve_side == 0:  # 左チームのサーブ
    cv2.circle(board, (left_x, serve_y), int(h * 0.03), (0, 0, 255), -1)
  elif serve_side == 1:  # 右チームのサーブ
    cv2.circle(board, (right_x, serve_y), int(h * 0.03), (0, 0, 255), -1)
  else:
    cv2.circle(board, (int(w/2), serve_y), int(h * 0.03), (0, 0, 255), -1)

  return board


def auto_pointing(pose_history, left_point, right_point, serve_side):

  if len(pose_history) != 0:
    if pose_history[0] == 0:
      left_point += 1
      serve_side = 0
    elif pose_history[0] == 1:
      right_point += 1
      serve_side = 1
    elif pose_history[0] == 9:
      serve_side = -1
    else:
      return left_point, right_point, serve_side

  if serve_side == 0:
    pointing_save_csv(pose_history, left_point, None)
  elif serve_side == 1:
    pointing_save_csv(pose_history, None, right_point)
  else:
    pointing_save_csv(None, left_point, right_point)

  return left_point, right_point, serve_side


def pointing_save_csv(pose_history, left_point, right_point):
  csv_file = pointing_csv_path
  headers = ["left_pose_history", "left_point", "right_point", "right_pose_history"]

  if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
      writer = csv.writer(file)
      writer.writerow(headers)
      writer.writerow(["", 0, 0, ""])

  if pose_history is not None:
    if left_point is not None:  # 左チーム得点
      row = [pose_history, left_point, "", ""]
    elif right_point is not None:  # 右チーム得点
      row = ["", "", right_point, pose_history]
    else:
      row = ["", "", "", ""]
  else:
    # pose_historyがNoneの場合は両方の得点のみを記録
    row = ["", left_point, right_point, ""]

  with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(row)




def get_combination_label(history):
  # 定義された組み合わせ辞書
  combination_dict = {
    (0, 2): "ball_touched",
    (0, 3): "ball_touched",
    (1, 2): "ball_touched",
    (1, 3): "ball_touched",
    (2, 0): "left_team_timeout",
    (3, 0): "left_team_timeout",
    (2, 1): "right_team_timeout",
    (3, 1): "right_team_timeout",
    (0, 4): "left_team_serve_allowed",  # 左チームのサーブ許可
    (1, 5): "right_team_serve_allowed",  # 右チームのサーブ許可
    # 他の組み合わせをここに追加
  }

  # 組み合わせ辞書のキーを確認
  for key, label in combination_dict.items():
    # 組み合わせが一致する場合
    if tuple(history[-len(key):]) == key:
      # 組み合わせを削除
      # history = history[:-len(key)]
      # 更新したhistoryと一致したlabelを返す
      return history, label

  # 該当なしの場合、historyをそのまま返しlabelはNone
  return history, "no action"



def update_class_queue(judge_class_queue, new_class, M):
    
  if judge_class_queue == 0:
    pose_queue.append(new_class)
    most_class, queue = class_counts(pose_queue, M)
  elif judge_class_queue == 1:
    right_queue.append(new_class)
    most_class, queue = class_counts(right_queue, M)
  else:
    left_queue.append(new_class)
    most_class, queue = class_counts(left_queue, M)

  return most_class, queue

    

def class_counts(queue, M):    
  # キュー内のクラス頻度を計算
  class_counts = Counter(queue)
  
  # 最頻クラスとその出現数を取得
  most_common_class, count = class_counts.most_common(1)[0]
  
  # M個以上出現していればそのクラスを返す
  if count >= M:
    return most_common_class, list(queue)
  else:
    return None, list(queue)


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
  return label_dict.get(class_index, None)  # 該当がない場合 "Unknown" を返す  


def get_hands_label(class_index):
  label_dict = {
    0:"scissors(double touch)",
    1:"four(four touch)",
    2:"three",
    3:"good",
    4:"other",
    5:"one"
    }
  return label_dict.get(class_index, "other")  # 該当がない場合 "Unknown" を返す

def draw_pose_class(image, pose_history):

  y_offset = 50

  for pose_id in pose_history:
    label = get_pose_label(pose_id)
    cv2.putText(image, label, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    y_offset += 40
    cv2.putText(image, "v", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    y_offset += 40

  # for pose in pose_history:
  #   cv2.putText(image, pose, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
  #   y_offset += 40
  #   cv2.putText(image, "v", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
  #   y_offset += 40
  
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
    # for hand_landmarks in landmarks:  # 各手のランドマークを個別に処理
        for idx in indices:
            landmark = landmarks.landmark[idx]
            # 必要な描画処理を実行
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, color, -1)


# def logging_csv(landmark_list, handedness):
#   # CSVファイルにデータを追記する
#   with open(output_csv_path, 'a', newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow([pose_label, handedness.classification[0].label[0:], *landmark_list])  # データ行を追記する


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