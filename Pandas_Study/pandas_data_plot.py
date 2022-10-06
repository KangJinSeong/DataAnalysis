'''
Date: 2022.10.05
Title: 
By: Kang Jin Seong
'''

import pandas as pd
import matplotlib.pyplot as plt

file_path = 'C:/Users/Kyngwon_SP/Desktop/Python/5674-833_4th/part4/시도별 전출입 인구수.xlsx'

df = pd.read_excel(file_path, engine= 'openpyxl', header= 0)
# print(df)
# print(df.info())

# 누락값(NAN)을 앞 데이터로 채움
df = df.fillna(method = 'ffill')

# 서울에서 다른 지역으로 이동한 데이터만 추출하여 정리
mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis = 1)
df_seoul.rename({'전입지별':'전입지'}, axis = 1, inplace= True)
df_seoul.set_index('전입지', inplace= True)
# print(df_seoul)


'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   '''
sr_one = df_seoul.loc['경기도']
# plt.figure(figsize = (14, 5))
# plt.plot(sr_one.index, sr_one.values)
# plt.title('seoul -> gungi people movement')
# plt.xlabel('year');plt.ylabel('movement')
# plt.xticks(rotation = 'vertical')
# plt.show()

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   '''
col_years = list(map(str, range(1970, 2018)))
df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]
df_4 = df_4.transpose()

# plt.style.use('ggplot')

df_4.index = df_4.index.map(int)

# df_4.plot(kind = 'area', stacked=False, alpha = 0.2, figsize= (20,10))
# plt.legend(loc = 'best', fontsize = 15)
# plt.show()

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

file_path = 'C:/Users/Kyngwon_SP/Desktop/Python/5674-833_4th/part4/남북한발전전력량.xlsx'
# plt.style.use('ggplot') # 스타일 서식 지정
# plt.rcParams['axes.unicode_minus']  = False # 마이너스 부호 출력 설정

# 엑셀 데이터 -> 데이터 프레임으로 변환
df = pd.read_excel(file_path, engine= 'openpyxl', convert_float = True)
df = df.loc[5:9]

df.drop('전력량 (억㎾h)', axis = 1, inplace = True)
df.set_index('발전 전력별', inplace = True)
df = df.T

# 증감률 계산
df = df.rename(columns = {'합계': '총발전량'})
df['총발전량-1년'] = df['총발전량'].shift(1)
df['증감률'] = ((df['총발전량'] / df['총발전량-1년']) - 1) * 100

# 2차축 그래프 그리기
# ax1  = df[['수력', '화력']].plot(kind = 'bar', figsize = (20,10), width = 0.7, stacked = True)
# ax2 = ax1.twinx()
# ax2.plot(df.index, df.증감률, ls = '--', marker = 'o', markersize = 20, color = 'red', label = '전년대비 증감률(%)')

# ax1.set_ylim(0, 500)
# ax2.set_ylim(-50, 50)
# # plt.show()
# print(df)

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
# plt.style.use('default')

file_path = file_path = 'C:/Users/Kyngwon_SP/Desktop/Python/5674-833_4th/part4/auto-mpg.csv'
df = pd.read_csv(file_path, header= 0)
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration','model year', 'origin', 'name']
cylinders_size = df.cylinders / df.cylinders.max() * 300
# df.plot(kind = 'scatter', x = 'weight', y = 'mpg', c = 'coral', figsize= (10,5),
#         s = cylinders_size)
# plt.title('Scatter Plot : mpg-weight-cylinders')
# plt.show()
# df.plot(kind = 'scatter', x = 'weight', y = 'mpg', marker = '+', cmap = 'viridis', figsize= (10,5),
#         c = cylinders_size, s = 50, alpha = 0.3)

# plt.savefig('scatter.png')
# plt.savefig('scatter_transparent.png', transparent = True)
# plt.show()

# 데이터 개수 카운터를 위해 값 1을 가진 열 추가
df['count'] = 1
df_origin = df.groupby('origin').sum()  # origin 열을 기준으로 그룹화 합계 연산
# print(df_origin.head())

# 제조국가 값을 실제 지역명으로 변경
df_origin.index = ['USA', 'EU', 'JPN']
# df_origin['count'].plot(kind = 'pie', figsize=(7,5), autopct = '%1.1f%%',
#                         startangle = 10, colors = ['chocolate', 'bisque', 'cadetblue'])
# plt.title('Model Origin', size = 20)
# plt.axis('equal')   # 파이 차트의 비율을 같게 조정
# plt.legend( labels = df_origin.index, loc = 'upper right')
# plt.show()

# plt.style.use('seaborn-poster')
# plt.rcParams['axes.unicode_minus'] = False
# df.drop('count',axis = 1, inplace=True)
# fig = plt.figure(figsize=(15, 5))
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)

# ax1.boxplot(x = [df[df['origin'] == 1]['mpg'],
#             df[df['origin'] == 2]['mpg'],
#             df[df['origin'] == 3]['mpg']],
#             labels = ['USA', 'EU', 'JAPAN'])

# ax2.boxplot(x = [df[df['origin'] == 1]['mpg'],
#             df[df['origin'] == 2]['mpg'],
#             df[df['origin'] == 3]['mpg']],
#             labels = ['USA', 'EU', 'JAPAN'],
#             vert= False)
# plt.show()
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
'''
Date: 2022.10.06
Title: 
By: Kang Jin Seong
'''
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

import seaborn as sns
import matplotlib.pyplot as plt
# 타이타닉 데이터 셋 가져오기
titanic = sns.load_dataset('titanic')
print(titanic.head());print('\n')
print(titanic.info())

# 회귀선이 있는 산점도
# 스타일 있는 테마 설정(darkgrid, wihtegrid, dark, white, ticks)
# sns.set_style('darkgrid')
# fig = plt.figure(figsize=(15,5))
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)

# # 그래프 그리기 선형회귀선 표시
# sns.regplot(x = 'age', y = 'fare', data = titanic, ax = ax1)

# sns.regplot(x = 'age', y = 'fare', data = titanic, ax = ax2, fit_reg = False)
# plt.show()

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
# 히스트 밀도 함수
# fig = plt.figure(figsize=(15,5))
# ax1 = fig.add_subplot(1,3,1)
# ax2 = fig.add_subplot(1,3,2)
# ax3 = fig.add_subplot(1,3,3)

# # displot
# sns.distplot(titanic['fare'], ax = ax1)
# # kdeplot
# sns.kdeplot(x = 'fare', data = titanic, ax = ax2)
# #histplo
# sns.histplot(x = 'fare', data = titanic, ax = ax3)

# plt.show()

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

# 히트맵
# 피벗테이블로 범주형 변수를 각각 행, 열로 재구분하여 정리
# table = titanic.pivot_table(index = ['sex'], columns = ['class'], aggfunc = 'size')

# # 히트맵 그리기
# sns.heatmap(table, annot = True, fmt = 'd', cmap = 'YlGnBu', linewidths= .5, cbar = False)
# plt.show()

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
# 범주형 데이터의 산점도

# sns.set_style('whitegrid')
# fig = plt.figure(figsize=(15,5))
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)

# # 이산형 변수의 분포 - 데이터 분산 미고려(중복 표시 0)
# sns.stripplot(x = 'class', y = 'age', data = titanic, ax = ax1)#, hue = 'sex')

# # 데이터 분산 고려(중복 표시 X)
# sns.swarmplot(x = 'class', y = 'age', data  = titanic, ax = ax2)#, hue = 'sex')

# plt.show()
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

# #막대 그래프
# fig = plt.figure(figsize = (15, 5))
# ax1 = fig.add_subplot(1,3,1)
# ax2 = fig.add_subplot(1,3,2)
# ax3 = fig.add_subplot(1,3,3)

# sns.barplot(x = 'sex', y = 'survived', data = titanic, ax = ax1)

# # hue 옵션 추가
# sns.barplot(x = 'sex', y = 'survived', data = titanic, ax = ax2, hue = 'class')
# # hue 누적 출력
# sns.barplot(x = 'sex', y = 'survived', data = titanic, ax = ax3, hue = 'class', dodge = False)

# ax1.set_title('titanic survived - sex')
# ax2.set_title('titanic survived - sex/class')
# ax3.set_title('titanic survived - sex/class(stacked)')
# plt.show()
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
# 빈도 그래프

# fig = plt.figure(figsize = (15, 5))
# ax1 = fig.add_subplot(1,3,1)
# ax2 = fig.add_subplot(1,3,2)
# ax3 = fig.add_subplot(1,3,3)
# sns.countplot(x = 'class', palette = 'Set1', data = titanic, ax = ax1)

# # hue 옵션에 'who'추가
# sns.countplot(x = 'class', hue = 'who', palette = 'Set2', data = titanic, ax = ax2)

# #dodge = False 옵션 추가(축방향으로 분리하지 않고 누적 그래프 출력)
# sns.countplot(x = 'class', hue = 'who', palette = 'Set3', dodge = False, data = titanic, ax = ax3)

# ax1.set_title('titanic class')
# ax2.set_title('titanic class = who')
# ax3.set_title('titanic class - who(stacked)')
# plt.show()
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
# 박스 플롯/바이올린 그래프
# sns.set_style('whitegrid')
# fig = plt.figure(figsize=(15,10))
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(2,2,2)
# ax3 = fig.add_subplot(2,2,3)
# ax4 = fig.add_subplot(2,2,4)

# # 박스 플롯 기본값
# sns.boxplot(x = 'alive', y = 'age', data = titanic, ax = ax1)
# # 박스 그래프- hue 변수 추가
# sns.boxplot(x = 'alive', y = 'age', hue = 'sex', data = titanic, ax = ax2)
# # 바이올린플롯 기본값
# sns.violinplot(x = 'alive', y = 'age', data = titanic, ax = ax3)
# sns.violinplot(x = 'alive', y = 'age', hue = 'sex', data = titanic, ax = ax4)
# plt.show()

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
# # 조인트 그래프
# sns.set_style('whitegrid')

# # 산점도
# j1 = sns.jointplot(x = 'fare', y = 'age', data=titanic)

# # 회귀선
# j2 = sns.jointplot(x = 'fare', y = 'age', data=titanic, kind = 'reg')

# #육각 그래프
# j3 = sns.jointplot(x = 'fare', y = 'age', data=titanic, kind = 'hex')

# # 커널 밀집
# j4 = sns.jointplot(x = 'fare', y = 'age', data=titanic, kind = 'kde')

# j1.fig.suptitle('titanic fare - scatter', size = 15)
# j2.fig.suptitle('titanic fare - reg', size = 15)
# j3.fig.suptitle('titanic fare - hex', size = 15)
# j4.fig.suptitle('titanic fare - kde', size = 15)
# plt.show()

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
# 조건에 맞게 화면 분할
sns.set_style('whitegrid')

# 조건에 따라 그리드 나누기
# g = sns.FacetGrid(data = titanic, col = 'who', row = 'survived')
# # 그래프 적용하기 
# g = g.map(plt.hist, 'age')

# 이변수 데이터의 분포
titanic_pair = titanic[['age', 'pclass', 'fare']]

g = sns.pairplot(titanic_pair)



plt.show()