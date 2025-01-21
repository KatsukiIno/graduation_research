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
import utilities.utility as util

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# input_video_path = "input/videos/hand_videos/one.mp4"
input_video_path = "input/videos/action_videos/8_both_hands_raising/10.mp4"
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

            brect = util.calc_bounding_rect(debug_image, hand_landmarks)
            
            hand_landmark_list = util.calc_hand_landmark_list(debug_image, hand_landmarks, hand_target_indices)
            pre_processed_hand_landmark = util.pre_process_landmark(hand_landmark_list)
            util.draw_specific_hand_landmarks(debug_image, hands_results.multi_hand_landmarks, hand_target_indices, (0, 0, 255), handedness)
            debug_image = util.draw_bounding_rect(debug_image, brect)
            debug_image = util.draw_info_text(debug_image, brect, handedness)

            hand_input_data = np.array([pre_processed_hand_landmark], dtype=np.float32)
            hand_probs = util.run_model(interpreter_hand, hand_input_data)
            right_probs = None
            left_probs = None

            if handedness.classification[0].label == "Right":
              right_probs = hand_probs
            else:
              left_probs = hand_probs

            if mode:
              util.logging_csv(pre_processed_hand_landmark, handedness)
            else:
              hands_display_image = util.draw_hands_class_probabilities(hands_display_image, right_probs, left_probs)


      if pose_results.pose_landmarks is not None:

        brect = util.calc_bounding_rect(debug_image, pose_results.pose_landmarks)
        pose_landmarks = util.calc_landmark_list(debug_image, pose_results.pose_landmarks, pose_target_indices)
        pre_processed_pose_landmark = util.pre_process_landmark(pose_landmarks)
        util.draw_specific_pose_landmarks(debug_image, pose_results.pose_landmarks, pose_target_indices, (0, 255, 0))
        debug_image = util.draw_bounding_rect(debug_image, brect)

        pose_input_data = np.array([pre_processed_pose_landmark], dtype=np.float32)
        pose_probs = util.run_model(interpreter_pose, pose_input_data)

        # pose_history.append(get_pose_label(np.argmax(pose_probs)))

        debug_image = util.draw_info_text(debug_image, brect, None)

        # ///////////////////////////////////////////////////////
        class_label, class_queue = util.update_class_queue(class_queue, np.argmax(pose_probs), 5)
        if prov_class_label != class_label:
          print(f"最頻クラス：{util.get_pose_label(class_label)}")
          print(f"クラスキュー：{class_queue}")
          print()
        prov_class_label = class_label
        # ///////////////////////////////////////////////////////

        if mode:
          util.logging_csv(output_csv_path, pre_processed_pose_landmark, pose_label, handedness)
        else:
          pose_display_image = util.draw_pose_class_probabilities(pose_display_image, pose_probs)
      else:
        pose_probs = None

      frame_count += 1
      
      cv2.putText(debug_image, f"frame_count:{frame_count} / {total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.putText(debug_image, f"Mode: {'ON' if mode else 'OFF'}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if mode else (0, 0, 255), 2, cv2.LINE_AA)
      cv2.rectangle(debug_image, (40, 310), (300, 360), (128, 128, 128), -1)
      cv2.putText(debug_image, f"{util.get_pose_label(class_label)}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
      
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

if __name__ == "__main__":
  main()