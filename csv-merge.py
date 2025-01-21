# import pandas as pd

# # ファイルパス
# pose_csv_path = 'model/pose_classifier'
# output_csv_path = 'model/pose_classifier/merged_data_sample.csv'

# # 各CSVファイルの読み込みとラベルの追加
# df_noPose = pd.read_csv(pose_csv_path + '/noPose_sample.csv', header=None)

# df_out = pd.read_csv(pose_csv_path + '/out_sample.csv', header=None)
# # df_out['label'] = 0  # ポーズ無し = 0

# df_right_hand_piece = pd.read_csv(pose_csv_path + '/right_hand_piece_sample.csv', header=None)
# # df_right_hand_piece['label'] = 1  # right_hand_piece = 1

# df_left_hand_piece = pd.read_csv(pose_csv_path + '/left_hand_piece_sample.csv', header=None)
# # df_left_hand_piece['label'] = 2  # left_hand_piece = 2

# # df_out = pd.read_csv(pose_csv_path + '/out_test.csv')
# # df_out['label'] = 3  # out = 3
# df_T1 = pd.read_csv(pose_csv_path + '/T1_sample.csv', header=None)

# df_T2 = pd.read_csv(pose_csv_path + '/T2_sample.csv', header=None)

# df_right_hand_four = pd.read_csv(pose_csv_path + '/right_hand_four_sample.csv', header=None)

# df_left_hand_four = pd.read_csv(pose_csv_path + '/left_hand_four_sample.csv', header=None)

# df_good = pd.read_csv(pose_csv_path + '/good.csv', header=None)

# df_hand_cross = pd.read_csv(pose_csv_path + '/hand_cross.csv', header=None)

# df_rihgt_90 = pd.read_csv(pose_csv_path + '/right_90.csv', header=None)

# df_right_hand_chest = pd.read_csv(pose_csv_path + '/right_hand_chest.csv', header=None)

# df_left_90 = pd.read_csv(pose_csv_path + '/left_90.csv', header=None)

# df_left_hand_chest = pd.read_csv(pose_csv_path + '/left_hand_chest.csv', header=None)

# # データの結合
# df = pd.concat([df_noPose, df_out, df_right_hand_piece, df_left_hand_piece, df_T1, df_T2, df_right_hand_four, df_left_hand_four, df_good, df_hand_cross, df_rihgt_90, df_right_hand_chest, df_left_90, df_left_hand_chest], ignore_index=True)

# # 結合したデータを新しいCSVファイルに保存
# df.to_csv(output_csv_path, index=False)

# print(f'結合したデータを {output_csv_path} に保存しました。')


import pandas as pd

# ファイルパス
pose_csv_path = 'model/pose_classifier'
output_csv_path = 'model/pose_classifier/merged_data_1.csv'

# 各CSVファイルの読み込みとラベルの追加
df_no_pose = pd.read_csv(pose_csv_path + '/no_pose.csv', header=None)

df_left_team_pointing = pd.read_csv(pose_csv_path + '/left_team_pointing.csv', header=None)

df_right_team_pointing = pd.read_csv(pose_csv_path + '/right_team_pointing.csv', header=None)

df_left_serve_allowed = pd.read_csv(pose_csv_path + '/left_serve_allowed.csv', header=None)

df_right_serve_allowed = pd.read_csv(pose_csv_path + '/right_serve_allowed.csv', header=None)

df_left_timeout = pd.read_csv(pose_csv_path + '/left_timeout.csv', header=None)

df_right_timeout = pd.read_csv(pose_csv_path + '/right_timeout.csv', header=None)

df_double_contact = pd.read_csv(pose_csv_path + '/double_contact.csv', header=None)

df_four_hit = pd.read_csv(pose_csv_path + '/four_hit.csv', header=None)

df_catched_ball = pd.read_csv(pose_csv_path + '/catched_ball.csv', header=None)

df_end_of_set = pd.read_csv(pose_csv_path + '/end_of_set.csv', header=None)

df_change_of_courts = pd.read_csv(pose_csv_path + '/change_of_courts.csv', header=None)

df_substitutions = pd.read_csv(pose_csv_path + '/substitutions.csv', header=None)

df_touch_net = pd.read_csv(pose_csv_path + '/touch_net.csv', header=None)

df_attack_hit_fault = pd.read_csv(pose_csv_path + '/attack_hit_fault.csv', header=None)

df_ball_touched = pd.read_csv(pose_csv_path + '/ball_touched.csv', header=None)

df_double_fault = pd.read_csv(pose_csv_path + '/double_fault.csv', header=None)

df_ball_out = pd.read_csv(pose_csv_path + '/ball_out.csv', header=None)

df_blocking_fault = pd.read_csv(pose_csv_path + '/blocking_fault.csv', header=None)

df_positional_fault = pd.read_csv(pose_csv_path + '/positional_fault.csv', header=None)

df_delay_in_service = pd.read_csv(pose_csv_path + '/delay_in_service.csv', header=None)

df_over_net = pd.read_csv(pose_csv_path + '/over_net.csv', header=None)

# データの結合
df = pd.concat([df_no_pose, df_left_team_pointing, df_right_team_pointing, df_left_serve_allowed, df_right_serve_allowed, df_left_timeout,
                df_right_timeout, df_double_contact, df_four_hit, df_catched_ball, df_end_of_set, df_change_of_courts, df_substitutions,
                df_touch_net, df_attack_hit_fault, df_ball_touched, df_double_fault, df_ball_out, df_blocking_fault, df_positional_fault,
                df_delay_in_service, df_over_net], ignore_index=True)

# 結合したデータを新しいCSVファイルに保存
df.to_csv(output_csv_path, index=False, header=False)

print(f'結合したデータを {output_csv_path} に保存しました。')
