import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as train
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns 
from folium import Map
from folium.plugins import HeatMap


# 定义数据
data = pd.read_csv(r'assets\Motor_Vehicle\Motor_Vehicle_Collisions_-202122.csv') 
print(data)
data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'], format='%m/%d/%Y', errors='coerce')
data['CRASH TIME'] = pd.to_datetime(data['CRASH TIME'], format='%H:%M', errors='coerce').dt.hour

# 定义月份
data['CRASH MONTH'] = data['CRASH DATE'].dt.to_period('M')

# 删除无效数据（如经纬度为(0.0, 0.0)的行）
data = data[(data['LATITUDE'].notnull()) & (data['LONGITUDE'].notnull()) & (data['LATITUDE'] != 0.0) & (data['LONGITUDE'] != 0.0)]

# 填补缺失值(因素不完整)
data['BOROUGH'].fillna('UNKNOWN', inplace=True)
data['ZIP CODE'].fillna(0, inplace=True)
data['CONTRIBUTING FACTOR VEHICLE 1'].fillna('Unspecified', inplace=True)

# 添加新的特征
data['CRASH MONTH'] = data['CRASH DATE'].dt.month
data['CRASH YEAR'] = data['CRASH DATE'].dt.year

# 删除不作为特征输入的列
data.drop(columns=['LOCATION', 'ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME'], inplace=True)

# processed_data_path = r'assets\Motor_Vehicle\Motor_Vehicle.csv'
# data.to_csv(processed_data_path, index=False)

# # 独热编码
# data = pd.get_dummies(data, columns=['BOROUGH', 'CONTRIBUTING FACTOR VEHICLE 1'])

# 事故地点分布（散点图）
plt.figure(figsize=(12, 8))
sns.scatterplot(x='LONGITUDE', y='LATITUDE', data=data)
plt.title('Geographic Distribution of Accidents')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig(r"assets\save\散点图.png")
# plt.show()

# 事故地点分布（热力图）
m = Map(location=[data['LATITUDE'].mean(), data['LONGITUDE'].mean()], zoom_start=11)
heat_data = [[row['LATITUDE'], row['LONGITUDE']] for index, row in data.iterrows() if not pd.isnull(row['LATITUDE']) and not pd.isnull(row['LONGITUDE'])]
HeatMap(heat_data).add_to(m)
m.save('accident_heatmap.html')

# 时间分布（时间序列图）
plt.figure(figsize=(12, 6))
sns.countplot(x='CRASH TIME', data=data)
plt.title('Accidents by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.grid(True)
plt.savefig(r"assets\save\时间序列图_小时.png")
# plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='CRASH MONTH', data=data)
plt.title('Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
# plt.xticks(rotation=45) 
plt.grid(True)
plt.savefig(r"assets\save\时间序列图_月份.png")
# plt.show()


# 影响因素分析（堆积柱状图）
plt.figure(figsize=(32, 20))
factor_counts = data['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()
sns.barplot(x=factor_counts.index, y=factor_counts.values)
plt.title('Accidents by Contributing Factors')
plt.xlabel('Contributing Factors')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=90)
plt.grid(True)
plt.savefig(r"assets\save\堆积柱状.png")
# plt.show()

# 伤亡情况分析（箱线图）
rename_dict = {
    'NUMBER OF PERSONS INJURED': 'PERSONS INJURED',
    'NUMBER OF PERSONS KILLED': 'PERSONS KILLED',
    'NUMBER OF PEDESTRIANS INJURED': 'PEDESTRIANS INJURED',
    'NUMBER OF PEDESTRIANS KILLED': 'PEDESTRIANS KILLED',
    'NUMBER OF CYCLIST INJURED': 'CYCLIST INJURED',
    'NUMBER OF CYCLIST KILLED': 'CYCLIST KILLED',
    'NUMBER OF MOTORIST INJURED': 'MOTORIST INJURED',
    'NUMBER OF MOTORIST KILLED': 'MOTORIST KILLED'
}

data = data.rename(columns=rename_dict)
injury_columns = ['PERSONS INJURED', 'PERSONS KILLED', 'PEDESTRIANS INJURED', 'PEDESTRIANS KILLED', 'CYCLIST INJURED', 'CYCLIST KILLED', 'MOTORIST INJURED', 'MOTORIST KILLED']
plt.figure(figsize=(16, 10))
sns.boxplot(data=data[injury_columns])
plt.title('Injuries and Fatalities Distribution')
plt.xticks(rotation=90)
plt.xlabel('Type of Injury')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout() 
plt.savefig(r"assets\save\箱线图.png")
# plt.show()

# 相关性分析（相关性热图）
correlation_data = data[injury_columns]
plt.figure(figsize=(16, 16))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Accident Data')
plt.savefig(r"assets\save\相关性热图.png")
# plt.show()

