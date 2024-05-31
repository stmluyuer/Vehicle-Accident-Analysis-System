import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# 预测地点是否发生了事故，或者发生事故后具体受伤的人数。

model_path = 'assets/model/random_forest_model.joblib'
model = joblib.load(model_path)

num_samples = 30

latitude_range = (40.5, 40.9)  
longitude_range = (-74.0, -73.7)
crash_time_range = (0, 23)
crash_month_range = (1, 12)
crash_year_range = (2021, 2023)

test_data = pd.DataFrame({
    'LATITUDE': np.random.uniform(latitude_range[0], latitude_range[1], num_samples),
    'LONGITUDE': np.random.uniform(longitude_range[0], longitude_range[1], num_samples),
    'CRASH TIME': np.random.randint(crash_time_range[0], crash_time_range[1] + 1, num_samples),
    'CRASH MONTH': np.random.randint(crash_month_range[0], crash_month_range[1] + 1, num_samples),
    'CRASH YEAR': np.random.randint(crash_year_range[0], crash_year_range[1] + 1, num_samples),
})

# 添加独热编码的列，并假设所有因子和BOROUGH的值均为0，因为我们还不知道是否发生了事故，如果发生事故可以假设这些值，进一步预测人数
borough_columns = [col for col in model.feature_names_in_ if col.startswith('BOROUGH_')]
contrib_factor_1_columns = [col for col in model.feature_names_in_ if col.startswith('CONTRIBUTING FACTOR VEHICLE 1_')]
contrib_factor_2_columns = [col for col in model.feature_names_in_ if col.startswith('CONTRIBUTING FACTOR VEHICLE 2_')]

for col in borough_columns + contrib_factor_1_columns + contrib_factor_2_columns:
    test_data[col] = 0

# 预测
predictions = model.predict(test_data)
test_data['PREDICTED NUMBER OF PERSONS INJURED'] = np.round(predictions)

print(test_data)
