'''
Date: 2022.10.26
Title: 
By: Kang Jin Seong
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/market-price.csv'
bitcoin_df = pd.read_csv(file_path, names = ['day', 'price'])
# print(bitcoin_df.shape);print('\n')
# print(bitcoin_df.info());print('\n')
# print(bitcoin_df.tail())

# to_datetime으로 day 피처를 시계열 피처로 변환
bitcoin_df['day'] = pd.to_datetime(bitcoin_df['day'])
bitcoin_df.index = bitcoin_df['day']
bitcoin_df.set_index('day', inplace= True)
# print(bitcoin_df.head())

# bitcoin_df.plot()
# plt.show()

'''ARIMA 분석 (Autoregresiion Integrated Moving Average)'''
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
# print(bitcoin_df.price.values)

model = ARIMA(bitcoin_df.price.values, order = (2,1,2))
model_fit = model.fit()
# print(model_fit.summary())

# 데이터 시각화
# fig = model_fit.predict()
# plt.figure()
# plt.plot(fig[1:], label = 'forecast')
# plt.plot(bitcoin_df['price'].tolist()[1:], label ='y')
# plt.legend()

# residuals = pd.DataFrame(model_fit.resid[1:])
# residuals.plot()
# plt.show()

'''실제 데이터와 비교'''
forecast_data = model_fit.forecast(steps = 5)    # 학습 데이터셋으로부터 5일 뒤를 에측
forecast_data1 = model_fit.get_forecast(steps = 5)
# print(forecast_data1.summary_frame())
# 테스트 데이터셋을 불러옵니다.
test_file_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/market-price-test.csv'
bitcoin_test_df = pd.read_csv(test_file_path, names = ['ds', 'y'])
pred_y = forecast_data.tolist()  # 마지막 5일의 예측 값
test_y = bitcoin_test_df.y.values   # 실제 5일 가격 데이터
pred_y_lower = []
pred_y_upper = []

for lower in forecast_data1.summary_frame()['mean_ci_lower']:
    pred_y_lower.append(lower)

for upper in forecast_data1.summary_frame()['mean_ci_upper']:
    pred_y_upper.append(upper)

from matplotlib import font_manager

plt.rc('font', family='NanumGothic')

# plt.plot(pred_y, color = 'gold', label = '예측한 가격')
# plt.plot(pred_y_lower, color = 'red', label = '예측한 최저 가격')
# plt.plot(pred_y_upper, color = 'blue', label = '예측한 최고 가격')
# plt.plot(test_y, color = 'green', label = '실제 가격')
# plt.legend()
# plt.show()

# plt.plot(pred_y, color = 'gold')
# plt.plot(test_y, color = 'green')
# plt.show()

from fbprophet import Prophet
bitcoin_df = pd.read_csv(file_path, names = ['ds', 'y'])
prophet = Prophet(seasonality_mode = 'multiplicative',
                yearly_seasonality= True, weekly_seasonality= True,
                daily_seasonality= True,
                changepoint_prior_scale= 0.5)
prophet.fit(bitcoin_df)

# 5일을 내다보며 예측
future_data = prophet.make_future_dataframe(periods=5, freq = 'd')
forecast_data = prophet.predict(future_data)

# print(forecast_data.tail(5));print('\n')
# 데이터에 존재하지 않는 5일 단위의 미래를 예측한 값
# print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))

# fig1 = prophet.plot(forecast_data)

# fig2 = prophet.plot_components(forecast_data)

'''실제 데이터와 비교'''
bitcoin_test_df = pd.read_csv(test_file_path, names = ['ds', 'y'])

pred_y = forecast_data.yhat.values[-5:]
test_y = bitcoin_test_df.y.values
pred_y_lower = forecast_data.yhat_lower.values[-5:]
pred_y_upper = forecast_data.yhat_upper.values[-5:]

# plt.plot(pred_y, color = 'gold')
# plt.plot(pred_y_lower, color = 'red')
# plt.plot(pred_y_upper, color = 'blue')
# plt.plot(test_y, color = 'green')

# plt.plot()

'''활용: 더나은 결과를 위한 방법: 상한값 혹은 하한값'''
bitcoin_df = pd.read_csv(file_path, names = ['ds', 'y'])
bitcoin_df['cap'] = 20000

prophet = Prophet(seasonality_mode = 'multiplicative',
                yearly_seasonality= True, weekly_seasonality= True,
                growth = 'logistic',
                daily_seasonality= True,
                changepoint_prior_scale= 0.5)

prophet.fit(bitcoin_df)

future_data = prophet.make_future_dataframe(periods = 5, freq = 'd')

future_data['cap'] = 20000
forecast_data = prophet.predict(future_data)

fig = prophet.plot(forecast_data)
# %%
