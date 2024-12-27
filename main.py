import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# CSVファイルの読み込み
df = pd.read_csv('data/シミュレーション波形2.csv')

# X軸とY軸のデータを取得
x = df['Displacement (um)']
y = df['Load force / punch circumference (N/mm)']

# Xの値でグループ化し、Yの値の平均を計算
df_grouped = df.groupby('Displacement (um)')['Load force / punch circumference (N/mm)'].mean().reset_index()

# 新しいX軸とY軸のデータを取得
x_mean = df_grouped['Displacement (um)'].values
y_mean = df_grouped['Load force / punch circumference (N/mm)'].values

# グラフ全体の最大Y値とその時のX値を取得
max_y_index = np.argmax(y_mean) # 最大Y値のインデックスを取得
max_y = y_mean[max_y_index] # 最大Y値を取得
max_x = x_mean[max_y_index] # 最大Y値の時のX値を取得

# 立ち上がり位置の検出
first_y = y_mean[0]
threshold_y = first_y * 300
rising_index = np.argmax(y_mean > threshold_y)

# フィッティング用の関数 (F = kx)
def linear_func(x, k):
    return k * x

# 塑性領域のフィッティング関数(F = a*x + b*sqrt(x) + offset)
def plastic_func_with_offset(x, a, b, offset):
  return a * x + b * np.sqrt(x) + offset

# 結果を格納するリスト
all_x_fit = []
all_y_fit = []
all_labels=[]
all_r2=[]

# 弾性領域のフィッティング処理
fitting_points = 20 #初期値
best_r2 = -np.inf
best_k = None
best_x_fit = None
best_y_fit = None
bad_fit_count = 0
history_r2=[]

for i in range(1000): #安全のためイテレーション回数を制限
    # フィッティングする範囲を定義
    fit_x = x_mean[rising_index:rising_index + fitting_points]
    fit_y = y_mean[rising_index:rising_index + fitting_points]

    # フィッティング実行
    try:
        params, _ = curve_fit(linear_func, fit_x - fit_x[0], fit_y, p0=[1.0]) # 傾きの初期値は1.0とする
    except RuntimeError:
        print(f"弾性領域：最適化に失敗しました。{fitting_points}点でのフィッティングをスキップします")
        fitting_points+=1
        continue
    k = params[0]

    # フィッティング結果の計算
    y_fit = linear_func(fit_x-fit_x[0], k)

    # R2値を計算
    r2 = r2_score(fit_y, y_fit)
    history_r2.append(r2)
    

    # フィット率が悪化した場合の処理
    if r2 < best_r2:
        bad_fit_count += 1
        if bad_fit_count >= 5:
           print(f"弾性領域：フィット率が悪化し続けたため、{fitting_points - 5}点でのフィッティングを最終結果とします")
           best_r2=history_r2[-6]
           break

    else:
        best_r2 = r2
        best_k = k
        best_x_fit = fit_x
        best_y_fit = y_fit
        bad_fit_count = 0  # リセット

    fitting_points += 1
    if rising_index+fitting_points>=len(x_mean):
      print(f"弾性領域：フィッティング可能な範囲を超えました。{fitting_points - 1}点でのフィッティングを最終結果とします")
      break

# 弾性領域の最終点
elastic_end_index = rising_index + fitting_points - 1 # 最終点そのものを開始点にするため、-5 を削除

# 弾性領域の最終結果をリストに追加
all_x_fit.append(best_x_fit)
all_y_fit.append(best_y_fit)
all_labels.append(f'Elastic Fit (k={best_k:.2f})')
all_r2.append(best_r2)

# 塑性領域のフィッティングを繰り返す
current_start_index = elastic_end_index #初期値
plastic_region_count = 1 #塑性領域の繰り返し回数
while True: #塑性領域のフィッティングを繰り返す
    # current_start_indexがy_meanのサイズを超えていないかチェック
    if current_start_index >= len(y_mean):
        print(f"塑性領域：フィッティング可能な範囲を超えたため、フィッティングを終了します")
        break
    plastic_fitting_points = 10 #初期値
    best_plastic_r2 = -np.inf
    best_a = None
    best_b = None
    best_offset = None
    best_plastic_x_fit = None
    best_plastic_y_fit = None
    plastic_bad_fit_count = 0
    plastic_history_r2=[]
    max_x_reached = False


    for i in range(1000): #安全のためイテレーション回数を制限
        plastic_fit_x = x_mean[current_start_index:current_start_index + plastic_fitting_points]
        plastic_fit_y = y_mean[current_start_index:current_start_index + plastic_fitting_points]
        
        # offsetの取得前にcurrent_start_indexの範囲チェック
        if current_start_index < len(y_mean):
          # 塑性領域の開始点のY軸の値
            offset = y_mean[current_start_index]
        else :
          print("塑性領域：current_start_indexが範囲外のため、offsetの取得をスキップします")
          break

        # フィッティング実行
        try:
            params, _ = curve_fit(plastic_func_with_offset, plastic_fit_x - plastic_fit_x[0], plastic_fit_y, p0=[0.0, 1.0, offset]) # 初期値は適当
        except RuntimeError:
            print(f"塑性領域{plastic_region_count}：最適化に失敗しました。{plastic_fitting_points}点でのフィッティングをスキップします")
            plastic_fitting_points += 1
            continue
        a, b, offset_fit = params

        # フィッティング結果の計算
        plastic_y_fit = plastic_func_with_offset(plastic_fit_x - plastic_fit_x[0], a, b, offset_fit)

        # R2値を計算
        plastic_r2 = r2_score(plastic_fit_y, plastic_y_fit)
        plastic_history_r2.append(plastic_r2)


        # フィット率が悪化した場合の処理
        if plastic_r2 < best_plastic_r2:
            plastic_bad_fit_count += 1
            if plastic_bad_fit_count >= 15:
                print(f"塑性領域{plastic_region_count}：フィット率が悪化し続けたため、{plastic_fitting_points - 5}点でのフィッティングを終了します")
                best_plastic_r2=plastic_history_r2[-6]
                break
        else:
            best_plastic_r2 = plastic_r2
            best_a = a
            best_b = b
            best_offset = offset_fit
            best_plastic_x_fit = plastic_fit_x
            best_plastic_y_fit = plastic_y_fit
            plastic_bad_fit_count = 0  # リセット
        
        #最大値の時のX値を超えた場合は終了
        if any(x > max_x for x in plastic_fit_x):
          print(f"塑性領域{plastic_region_count}：最大Y値のX値({max_x:.2f})に到達したため、フィッティングを終了します")
          max_x_reached = True
          break
        
         # 全体のデータ範囲を超えた場合は、終了する
        if current_start_index + plastic_fitting_points >= len(x_mean):
            print(f"塑性領域{plastic_region_count}：フィッティング可能な範囲を超えたため、フィッティングを終了します")
            break

        plastic_fitting_points += 1

    
    #塑性領域の最終結果をリストに追加
    if best_a is not None:
      all_x_fit.append(best_plastic_x_fit)
      all_y_fit.append(best_plastic_y_fit)
      all_labels.append(f'Plastic Fit (a={best_a:.2f}, b={best_b:.2f}, offset={best_offset:.2f})')
      all_r2.append(best_plastic_r2)
      current_start_index = current_start_index + plastic_fitting_points-1 # 次の開始点を更新。-5を削除
      plastic_region_count+=1 #塑性領域のカウントを更新
    else:
      break # フィッティング結果がなかった場合は、ループを終了

    # 最大X値に到達したら、全体のループを終了
    if max_x_reached:
      break

# グラフの描画
plt.figure(figsize=(12, 7))
plt.plot(x_mean, y_mean, label='Average Load',linewidth=3)  # 元データをプロット
for i in range(len(all_x_fit)):
    if len(all_x_fit[i])>0:
      plt.plot(all_x_fit[i], all_y_fit[i], label=all_labels[i])

plt.xlabel('Displacement (um)')
plt.ylabel('Average Load force / punch circumference (N/mm)')
plt.title('Multi-Region Fitting')
plt.grid(True)
plt.legend()
plt.show()

#結果の表示
print(f"弾性領域の最終的な傾きk = {best_k:.2f}")
print(f"弾性領域の最終的なR2値 = {best_r2:.4f}")

for i in range(1, len(all_x_fit)):
  if len(all_x_fit[i]) > 0 and len(all_r2) > i:
    print(f"塑性領域{i}の最終的なパラメータ:{all_labels[i]}")
    print(f"塑性領域{i}の最終的なR2値 = {all_r2[i]:.4f}")