'''
Date: 2022.10.25
Title: 
By: Kang Jin Seong
'''

import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

picher_file_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/picher_stats_2017.csv'
batter_file_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/batter_stats_2017.csv'
picher = pd.read_csv(picher_file_path)
batter = pd.read_csv(batter_file_path)
picher.columns

# picher.head()

# picher.info()
# RangeIndex: 152 entries, 0 to 151
# Data columns (total 22 columns):
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   선수명       152 non-null    object 
#  1   팀명        152 non-null    object 
#  2   승         152 non-null    int64  
#  3   패         152 non-null    int64  
#  4   세         152 non-null    int64  
#  5   홀드        152 non-null    int64  
#  6   블론        152 non-null    int64  
#  7   경기        152 non-null    int64  
#  8   선발        152 non-null    int64  
#  9   이닝        152 non-null    float64
#  10  삼진/9      152 non-null    float64
#  11  볼넷/9      152 non-null    float64
#  12  홈런/9      152 non-null    float64
#  13  BABIP     152 non-null    float64
#  14  LOB%      152 non-null    float64
#  15  ERA       152 non-null    float64
#  16  RA9-WAR   152 non-null    float64
#  17  FIP       152 non-null    float64
#  18  kFIP      152 non-null    float64
#  19  WAR       152 non-null    float64
# ...
#  20  연봉(2018)  152 non-null    int64  
#  21  연봉(2017)  152 non-null    int64  
# dtypes: float64(11), int64(9), object(2)

# picher['연봉(2017)'].describe()

# picher['연봉(2018)'].hist(bins = 100)
# 
# picher.boxplot(column = ['연봉(2018)'])
# 
'''회귀 분석에 사용할 피처 살펴보기'''
picher_feature_df = picher[['승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', '볼넷/9', '홈런/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '연봉(2018)', '연봉(2017)']]

# 피처 각각에 대한 히스토그램을 출력합니다.
def plot_hist_each_column(df):
    plt.rcParams['figure.figsize'] = [20,16]
    fig = plt.figure(1)
    for i in range(len(df.columns)):
        ax = fig.add_subplot(5,5, i+1)
        plt.hist(df[df.columns[i]], bins = 50)
        ax.set_title(df.columns[i])
    plt.show()
# plot_hist_each_column(picher_feature_df)


# 
'''피처 스케일링'''
# 판다스 형태로 정의된 데이터를 출력할 때 scientific-notation이 아닌 float 모양으로 출력되게 해줍니다.
pd.options.mode.chained_assignment = None

# 피처 각각에 대한 스케일링을 수행하는 함수 정의
def standard_scaling(df, scale_columns):
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x: (x-series_mean)/series_std)
        return df

scale_columns = ['승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', '볼넷/9', '홈런/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '연봉(2018)', '연봉(2017)']
picher_df = standard_scaling(picher, scale_columns)
picher_df = picher_df.rename(columns = {'연봉(2018)' : 'y'})
# picher_df.head()
'''원- 핫 인코딩'''
team_encoding = pd.get_dummies(picher_df['팀명'])
picher_df = picher_df.drop('팀명', axis = 1)
picher_df = picher_df.join(team_encoding)
# picher_df.head(3)
# team_encoding.head()

'''회귀 분석을 위한 학습데이터 분리'''
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

X = picher_df[picher_df.columns.difference(['선수명', 'y'])]
# print(X);print('\n')
Y = picher_df['y']
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=19)

'''회귀 분석 계수 학습, 계수 출력'''
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
# print(lr.coef_)

'''어떤 피처가 영향력이 강한 피처일까'''
# import statsmodels.api as sm
# X_train = sm.add_constant(X_train)
# model = sm.OLS(y_train, X_train).fit()
# print(model.summary())
# # mpl.rc('font', family = 'NanumGOthicOTF')
# plt.rcParams['figure.figsize'] = [20, 16]

# coefs = model.params.tolist()
# coefs_series = pd.Series(coefs)

# x_labels = model.params.index.tolist()

# ax = coefs_series.plot(kind = 'bar')
# ax.set_title('feature_coef_graph')
# ax.set_xlabel('x_features')
# ax.set_ylabel('coef')
# ax.set_xticklabels(x_labels)
# plt.show()

'''피처들의 상관 관계 분석하기'''
import seaborn as sns
# print(scale_columns)
# scale_columns[scale_columns.index('연봉(2018)')] = 'y'
# corr = picher_df[scale_columns].corr(method= 'pearson')
# print(scale_columns)
# plt.rc('font', family = 'NanumGothicOTF')
# sns.set(font_scale = 1.5)
# hm = sns.heatmap(corr.values,
# cbar = True,
# annot = True,
# square = True,
# fmt = '.2f',
# annot_kws = {'size':15},
# yticklabels=scale_columns,
# xticklabels=scale_columns)

# plt.tight_layout()
# plt.show()

'''다중 공선성 확인'''
# 회귀 분석은 피처간의 독립성을 전제로 하는 분석
# 올바른 회귀 분석을 하려면 이러한 피처 쌍을 제거해야한다.

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['feature'] = X.columns
# print(vif.round(1))

'''시각화'''
X = picher_df[['FIP', 'WAR', '볼넷/9', '삼진/9','연봉(2017)']]
Y = picher_df['y']
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=19)

'''회귀 분석 계수 학습, 계수 출력'''
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

predict_2018_salary = lr.predict(X)
picher_df['예측연봉(2018)'] = pd.Series(predict_2018_salary)

picher = pd.read_csv(picher_file_path)
picher = picher[['선수명', '연봉(2017)']]

result_df = picher_df.sort_values(by = 'y', ascending= False)
result_df.drop(['연봉(2017)'], axis = 1, inplace = True, errors = 'ignore')
result_df = result_df.merge(picher, on = ['선수명'], how = 'left')
print(result_df.info())
result_df = result_df[['선수명', 'y', '예측연봉(2018)', '연봉(2017)']]
result_df.columns = ['선수명', '실제연봉(2018)', '예측연봉(2018)', '작년연봉(2017)']

# #재 계약하여 연봉이 변화한 선수만을 대상으로 관챃ㄹ
result_df = result_df[result_df['작년연봉(2017)'] != result_df['실제연봉(2018)']]
result_df = result_df.reset_index()
result_df = result_df.iloc[:10, :]
print(result_df.head(10))

result_df.plot(x = '선수명', y = ['작년연봉(2017)', '예측연봉(2018)', '실제연봉(2018)'], kind = 'bar')
plt.show()