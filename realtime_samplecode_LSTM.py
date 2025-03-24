# action 時系列動作分類　リアルタイム推論 pose+hand 修正必要handsが起動しない

import cv2
import copy
import itertools
import mediapipe as mp
import pandas as pd
import numpy as np
import csv
import os
import tensorflow as tf
from collections import deque, Counter

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

input_video_path = "input/videos/referee_action_videos/ball_touched_right.mp4"
# input_video_path = "input/videos/action_videos/2_left_ball_touch/2.mp4"
pointing_csv_path = 'output/csv/score_record/LSTM_scoring_sheet_1.csv'

pose_target_indices = {0} | set(range(11, 23))
hand_target_indices = [0, 4, 8, 12, 16, 20]

history_length = 30
past_positions_15 = deque(maxlen=history_length)
past_positions_16 = deque(maxlen=history_length)
point_history = deque(maxlen=history_length)

action_length = 10
pose_queue = deque(maxlen=action_length)
right_queue = deque(maxlen=action_length)
left_queue = deque(maxlen=action_length)

interpreter_hand = tf.lite.Interpreter(model_path="model/hand_classifier/hands_classifier_11_22.tflite")
interpreter_hand.allocate_tensors()

interpreter_pose = tf.lite.Interpreter(model_path="model/action_classifier/action_classifier_1.tflite")
interpreter_pose.allocate_tensors()

def main():

    cap = cv2.VideoCapture(input_video_path)
    frame_count = 1
    action_probs = None
    class_label = None
    prov_class_label = None
    prov_left_hand_class_label = None
    prov_right_hand_class_label = None
    pose_history = []
    left_hand_history = []
    right_hand_history = []
    most_common_class_left_hand = None 
    most_common_class_right_hand = None
    pose_threshold = 0

    left_team_point = 0
    right_team_point = 0
    serve_side = None

    count = 0

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose, \
    mp_hands.Hands(max_num_hands=2,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as hands:
    # with mp_holistic.Holistic(min_detection_confidence=0.5,
    #                           min_tracking_confidence=0.5) as holisitc:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            pose_results = pose.process(debug_image)
            image.flags.writeable = True

            action_class_display = np.ones((600, 600, 3), dtype=np.uint8) * 255  # 白背景の画像
            

            if pose_results.pose_landmarks is not None:
                brect = calc_bounding_rect(debug_image, pose_results.pose_landmarks)
                pose_landmarks = calc_landmark_list(debug_image, pose_results.pose_landmarks, pose_target_indices)
                pre_processed_pose_landmark = pre_process_landmark(pose_landmarks)
                draw_specific_pose_landmarks(debug_image, pose_results.pose_landmarks, pose_target_indices, (0, 0, 255))
                debug_image = draw_bounding_rect(debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, None)

                point_history.append(pre_processed_pose_landmark)

                if len(point_history) == history_length:
                  # pose_input_data = np.array([pre_processed_pose_landmark], dtype=np.float32)
                  pose_input_data = np.array(point_history).flatten().astype(np.float32).reshape(1, history_length, -1)
                  # print(pose_input_data.shape)
                  action_probs = run_model(interpreter_pose, pose_input_data)

                # point_history.append(pre_processed_pose_landmark)
                # past_positions_15.append(results.pose_landmarks.landmark[15])
                # past_positions_16.append(results.pose_landmarks.landmark[16])
                # debug_image = draw_past_positions(debug_image, past_positions_15, past_positions_16)
                frame_count += 1

                # ///////////////////////////////////////////////////////
                class_label, _ = update_class_queue(0, np.argmax(action_probs), pose_threshold)
                if class_label is not None:
                    if prov_class_label != class_label:
                        if prov_class_label == 11 and len(pose_history) > 2:  # 直前のクラスラベルが11なら
                            pose_history.clear()  # pose_historyをクリア
                            print(f"{count} / 10")
                            print("pose_history cleared")
                            print("")
                            count += 1

                        pose_history.append(class_label)
                        print(f"pose_history:{pose_history}")
                        prov_class_label = class_label
                        pose_history, _ = get_combination_label(pose_history)
                        if len(pose_history) > 2 and class_label == 11:
                          left_team_point, right_team_point, serve_side = auto_pointing(pose_history, left_team_point, right_team_point, serve_side)
                          if pose_history[0] == 9:
                            # break
                            continue
                # ///////////////////////////////////////////////////////

                action_class_display = draw_pose_class(action_class_display, pose_history)
            else:
                action_probs = None


            if pose_history and pose_history[-1] in [6, 7, 8]:
              hands_results = hands.process(debug_image)
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
                    right_hand_history.append(np.argmax(right_probs))
                    # if right_hand_class_label is not None:
                    #   if prov_right_hand_class_label != right_hand_class_label:
                    #     right_hand_history.append(right_hand_class_label)
                    #     print(f"right_hand:{right_hand_history}")
                    #     prov_right_hand_class_label = right_hand_class_label
                  else:
                    left_probs = hand_probs
                    left_hand_history.append(np.argmax(left_probs))
                    # left_hand_class_label, _ = update_class_queue(2, np.argmax(left_probs), 7)
                    # if left_hand_class_label is not None:
                    #   if prov_left_hand_class_label != left_hand_class_label:
                    #     left_hand_history.append(left_hand_class_label)
                    #     print(f"left_hand:{left_hand_history}")
                    #     prov_left_hand_class_label = left_hand_class_label
              else:
                if right_hand_history:
                  most_common_class_right_hand = Counter(right_hand_history).most_common(1)[0][0]
                if left_hand_history:
                  most_common_class_left_hand = Counter(left_hand_history).most_common(1)[0][0]
              
              if most_common_class_left_hand is not None or most_common_class_right_hand is not None:
                # print(f"right_hand:{most_common_class_right_hand}, left_hand:{most_common_class_left_hand}")
                hands_class = get_hands_class(most_common_class_left_hand, most_common_class_right_hand)
                print(f"{hands_class}: [{most_common_class_left_hand, most_common_class_right_hand}]")
                most_common_class_left_hand = None 
                most_common_class_right_hand = None
                # right_hand_history.clear()
                # left_hand_history.clear()
                # prov_left_hand_class_label = None
                # prov_right_hand_class_label = None
            # if pose_history:

            #   if pose_history[-1] in [6, 8]:
            #     hands_results = hands.process(image)
            #     if hands_results.right_hand_landmarks is not None:
            #       brect = calc_bounding_rect(debug_image, results.right_hand_landmarks)
                  
            #       hand_landmark_list = calc_hand_landmark_list(debug_image, results.right_hand_landmarks, hand_target_indices)
            #       pre_processed_hand_landmark = pre_process_landmark(hand_landmark_list)
            #       draw_specific_hand_landmarks(debug_image, results.right_hand_landmarks, hand_target_indices, (0, 0, 255), None)
            #       debug_image = draw_bounding_rect(debug_image, brect)
            #       debug_image = draw_info_text(debug_image, brect, None)

            #       hand_input_data = np.array([pre_processed_hand_landmark], dtype=np.float32)
            #       hand_probs = run_model(interpreter_hand, hand_input_data)
            #       right_probs = hand_probs

            #       # ////////////////////////////////////////////////
            #       right_hand_history.append(np.argmax(right_probs))
            #       # ////////////////////////////////////////////////

            #   else:
            #     if right_hand_history:
            #       most_common_class_right_hand = Counter(right_hand_history).most_common(1)[0][0]
            #       # print(f"右手最多クラス：{Counter(right_hand_history).most_common(1)[0][0]}，{get_hands_label(Counter(right_hand_history).most_common(1)[0][0])}")
            #       # print(right_hand_history)
            #       right_hand_history.clear()

            #   if pose_history[-1] in [7, 8]:
            #     if results.left_hand_landmarks is not None:
            #       brect = calc_bounding_rect(debug_image, results.left_hand_landmarks)
                  
            #       hand_landmark_list = calc_hand_landmark_list(debug_image, results.left_hand_landmarks, hand_target_indices)
            #       pre_processed_hand_landmark = pre_process_landmark(hand_landmark_list)
            #       draw_specific_hand_landmarks(debug_image, results.left_hand_landmarks, hand_target_indices, (0, 0, 255), None)
            #       debug_image = draw_bounding_rect(debug_image, brect)
            #       debug_image = draw_info_text(debug_image, brect, None)

            #       hand_input_data = np.array([pre_processed_hand_landmark], dtype=np.float32)
            #       hand_probs = run_model(interpreter_hand, hand_input_data)
            #       left_probs = hand_probs

            #       left_hand_history.append(np.argmax(left_probs))  
            #   else:
            #     if left_hand_history:
            #       most_common_class_left_hand = Counter(left_hand_history).most_common(1)[0][0]
            #       # print(f"左手最多クラス：{Counter(left_hand_history).most_common(1)[0][0]}，{get_hands_label(Counter(left_hand_history).most_common(1)[0][0])}")
            #       # print(left_hand_history)
            #       left_hand_history.clear()

            #   if most_common_class_left_hand is not None or most_common_class_right_hand is not None:
            #     # print(f"right_hand:{most_common_class_right_hand}, left_hand:{most_common_class_left_hand}")
            #     hands_class = get_hands_class(most_common_class_left_hand, most_common_class_right_hand)
            #     print(f"{hands_class}: {most_common_class_left_hand, most_common_class_right_hand}")
            #     most_common_class_left_hand = None 
            #     most_common_class_right_hand = None

            score_board = create_score_board(left_team_point, right_team_point, serve_side)

            cv2.putText(debug_image, 'frame_count : ' + str(frame_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('MediaPipe Holistic', debug_image)
            cv2.imshow('action class', action_class_display)
            cv2.imshow('score board', score_board)
            key = cv2.waitKey(5)
            if key & 0xFF == 27:  # ESCキーで終了
                break

    cap.release()
    cv2.destroyAllWindows()

# ///////////////////////////////////////////////////

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
  return history, None


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


def get_action_label(action_id):
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
        11:"chage_court"
        # 12:"other(no_pose)"
    }
    return label_dict.get(action_id, None)  # 該当がない場合 "Unknown" を返す  

def draw_pose_class(image, pose_history):

  y_offset = 50

  for pose_id in pose_history:
    label = get_action_label(pose_id)
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


def pointing_save_csv(pose_history, left_point, right_point):
  csv_file = pointing_csv_path
  headers = ["left_pose_history", "left_point", "right_point", "right_pose_history"]

  pose_history = [get_action_label(action) for action in pose_history] if pose_history else ""

  if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
      writer = csv.writer(file)
      writer.writerow(headers)
      writer.writerow(["", 0, 0, ""])

  if pose_history is not None:
    if left_point is not None:
      row = [pose_history, left_point, "", ""]
    elif right_point is not None:
      row = ["", "", right_point, pose_history]
    else:
      row = ["", "", "", ""]
  else:
    row = ["", left_point, right_point, ""]

  with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(row)


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
