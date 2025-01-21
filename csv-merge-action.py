import pandas as pd

# ファイルパス
action_csv_path = 'model/action_classifier'
output_csv_path = 'model/action_classifier/merged_data_yolo_ver.csv'

# 各CSVファイルの読み込みとラベルの追加
df_left_serve_allowed = pd.read_csv(action_csv_path + '/left_serve_allowed.csv', header=None)

df_right_serve_allowed = pd.read_csv(action_csv_path + '/right_serve_allowed.csv', header=None)

df_left_timeout = pd.read_csv(action_csv_path + '/left_timeout.csv', header=None)

df_right_timeout = pd.read_csv(action_csv_path + '/right_timeout.csv', header=None)

df_attack_hit_fault = pd.read_csv(action_csv_path + '/attack_hit_fault.csv', header=None)

df_catched_ball = pd.read_csv(action_csv_path + '/catched_ball.csv', header=None)

df_court_change = pd.read_csv(action_csv_path + '/court_change.csv', header=None)

df_substitutions = pd.read_csv(action_csv_path + '/substitutions.csv', header=None)

df_ball_contact = pd.read_csv(action_csv_path + '/ball_contact.csv', header=None)

df_no_pose = pd.read_csv(action_csv_path + '/no_pose.csv', header=None)

# データの結合
df = pd.concat([df_no_pose, df_left_serve_allowed, df_right_serve_allowed, df_left_timeout, df_right_timeout, df_attack_hit_fault, df_catched_ball,
                df_court_change, df_substitutions, df_ball_contact], ignore_index=True)

# 結合したデータを新しいCSVファイルに保存
df.to_csv(output_csv_path, index=False, header=False)

print(f'結合したデータを {output_csv_path} に保存しました。')
