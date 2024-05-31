import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from folium import Map
from folium.plugins import HeatMap
import joblib
from sklearn.utils import compute_sample_weight

# 定义数据
data = pd.read_csv(r'assets\Motor_Vehicle\Motor_Vehicle_Collisions_-202122.csv') 
print(data)

# 处理日期和时间数据
data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'], format='%m/%d/%Y', errors='coerce')
data['CRASH TIME'] = pd.to_datetime(data['CRASH TIME'], format='%H:%M', errors='coerce').dt.hour

# 删除无效数据（如经纬度为空或为(0.0, 0.0)的行）
data = data[(data['LATITUDE'].notnull()) & (data['LONGITUDE'].notnull()) & (data['LATITUDE'] != 0.0) & (data['LONGITUDE'] != 0.0)]

# 检查并删除重复行
data.drop_duplicates(inplace=True)

# 处理异常值
# data = data[(data['LATITUDE'] > 25) & (data['LATITUDE'] < 50)]
# data = data[(data['LONGITUDE'] > -130) & (data['LONGITUDE'] < -60)]

injuries_mean = data['NUMBER OF PERSONS INJURED'].mean()
injuries_std = data['NUMBER OF PERSONS INJURED'].std()
upper_limit = injuries_mean + 3 * injuries_std
lower_limit = injuries_mean - 3 * injuries_std
data['NUMBER OF PERSONS INJURED'] = np.where(data['NUMBER OF PERSONS INJURED'] > upper_limit, upper_limit,
                                             np.where(data['NUMBER OF PERSONS INJURED'] < lower_limit, lower_limit, data['NUMBER OF PERSONS INJURED']))

# 填补缺失值
data['BOROUGH'].fillna('UNKNOWN', inplace=True)
data['ZIP CODE'].fillna(0, inplace=True)
data['CONTRIBUTING FACTOR VEHICLE 1'].fillna('Unspecified', inplace=True)
data['CONTRIBUTING FACTOR VEHICLE 2'].fillna('Unspecified', inplace=True)

# 添加新的特征
data['CRASH MONTH'] = data['CRASH DATE'].dt.month
data['CRASH YEAR'] = data['CRASH DATE'].dt.year

# 独热编码
data = pd.get_dummies(data, columns=['BOROUGH', 'CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2'], dtype=int)
print(data)

# 特征选择
features = [
    'LATITUDE', 
    'LONGITUDE', 
    'CRASH TIME', 
    'CRASH MONTH', 
    'CRASH YEAR',
] + [col for col in data.columns if col.startswith('BOROUGH_') or col.startswith('CONTRIBUTING FACTOR VEHICLE 1_') or col.startswith('CONTRIBUTING FACTOR VEHICLE 2_')]
print(features)

# 导出独热编码后的数据框为 CSV 文件
feature_data = data[features]
features_data_path = 'selected_features_data.csv'
feature_data.to_csv(features_data_path, index=False)

X = data[features]
y = data['NUMBER OF PERSONS INJURED']

# 确保所有特征为数值类型，并填补缺失值
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = y.fillna(0)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 计算样本权重
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 训练随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights)



# 预测和评估
y_pred = model.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))

# # 将预测结果四舍五入到最近的整数
# y_pred_rounded = np.round(y_pred)

# 保存模型
model_path = 'assets/model/random_forest_model.joblib'
joblib.dump(model, model_path)

# 将预测结果四舍五入到最近的整数
y_pred_rounded = np.round(y_pred)

print('MSE:', mean_squared_error(y_test, y_pred_rounded))

# 计算全局准确率
tolerance = 1 
accuracy = np.mean(np.abs(y_test - y_pred_rounded) <= tolerance)
print(f'Accuracy within ±{tolerance}: {accuracy * 100:.2f}%')


# 确保样本索引在范围内
y_test_sample = y_test.sample(n=30, random_state=42).sort_index()
y_pred_sample = pd.Series(y_pred_rounded, index=y_test.index).loc[y_test_sample.index]

# 绘制实际值和预测值的散点图
plt.figure(figsize=(28, 14))
plt.scatter(y_test_sample.index, y_test_sample, label='Actual Injuries', alpha=0.6)
plt.scatter(y_test_sample.index, y_pred_sample, label='Predicted Injuries', alpha=0.6)
plt.xlabel('Index')
plt.ylabel('Number of Injuries')
plt.title('Actual vs Predicted Injuries (Sampled)')
plt.legend()
plt.grid(True)
# plt.savefig('assets/save/actual_vs_predicted_injuries_line_plot_1.png')
plt.show()

# 绘制实际值和预测值的曲线图
plt.figure(figsize=(28, 14))
plt.plot(y_test_sample.index, y_test_sample, label='Actual Injuries', alpha=0.6, marker='o')
plt.plot(y_test_sample.index, y_pred_sample, label='Predicted Injuries', alpha=0.6, marker='x')
plt.xlabel('Index')
plt.ylabel('Number of Injuries')
plt.title('Actual vs Predicted Injuries (Sampled)')
plt.legend()
plt.grid(True)
# plt.savefig('assets/save/actual_vs_predicted_injuries_line_plot_2.png')
plt.show()
