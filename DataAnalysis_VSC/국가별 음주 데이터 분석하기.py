'''
Date: 2022.10.20
Title: 
By: Kang Jin Seong
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/drinks.csv'

drinks = pd.read_csv(file_path)
print(drinks.info())
print(drinks.head())

#  #   Column                        Non-Null Count  Dtype
# ---  ------                        --------------  -----
#  0   country                       193 non-null    object 국가정보
#  1   beer_servings                 193 non-null    int64  
#  2   spirit_servings               193 non-null    int64
#  3   wine_servings                 193 non-null    int64
#  4   total_litres_of_pure_alcohol  193 non-null    float64
#  5   continent                     170 non-null    object 국가의 대륙정보
# %%
drinks.describe()
# %%
# 두 피처의 상관계수 구하기
# pearson은 상관 계수를 구하는 방법 중 하나
corr = drinks[['beer_servings', 'wine_servings']].corr(method='pearson')
print(corr)
# %%
# 다수의 피처 상관계수 구하기
corr = drinks[['beer_servings', 'wine_servings',
                'spirit_servings', 'total_litres_of_pure_alcohol']].corr(method = 'pearson')
print(corr)
# %%
# 상관계수 시각화 (heatmap, pairpol)
import seaborn as sns
cols_view = ['beer', 'wine', 'spirit', 'alcohol']
sns.set(font_scale = 1.5)
hm = sns.heatmap(corr.values,
                cbar = True,
                annot = True,
                square = True,
                fmt = '.2f',
                annot_kws = {'size':15},
                yticklabels = cols_view,
                xticklabels = cols_view)

plt.tight_layout()
plt.show()

# 산점도 그래프
sns.set(style = 'whitegrid', context = 'notebook')
sns.pairplot(drinks[['beer_servings', 'wine_servings',
                'spirit_servings', 'total_litres_of_pure_alcohol']], height = 2.5)
plt.show()
# %%
# 결측데이터 전처리
drinks['continent'] = drinks['continent'].fillna('OT')
drinks.head(10)
# %%
# 파이차트로 시각화 하기
labels = drinks['continent'].value_counts().index.tolist()
# print(labels)
fracs1 = drinks['continent'].value_counts().values.tolist()
explode = (0,0,0,0.25,0,0)

plt.pie(fracs1, explode=explode, labels = labels, autopct = '%.0f%%', shadow = True)
plt.show()
# %%
# agg() 함수를 이용해 대륙별로 분석하기
result = drinks.groupby('continent').spirit_servings.agg(['mean', 'min', 'max', 'sum'])
result.head()
# %%
# 전체 평균보다 많은 알콜올을 섭취하는 대륙을 구하라.
continent_mean = drinks.groupby('continent')['total_litres_of_pure_alcohol'].mean()
total_mean = drinks.total_litres_of_pure_alcohol.mean()
continent_over_mean = continent_mean[ continent_mean >= total_mean]
print(continent_over_mean)
# %%
# 평균 beer 소비량이 가장 높은 대륙은 어디일까?
beer_continent = drinks.groupby('continent').beer_servings.mean().idxmax()
print(beer_continent)
# %%
# 결과 시각화
n_groups = len(result.index)
means = result['mean'].tolist()
mins = result['min'].tolist()
maxs = result['max'].tolist()
sums = result['sum'].tolist()

index = np.arange(n_groups)
bar_width = 0.1

rects1 = plt.bar(index, means, bar_width, color = 'r', label = 'Mean')
rects2 = plt.bar(index + bar_width, mins, bar_width, color = 'g', label = 'Min')
rects3 = plt.bar(index + bar_width * 2 , maxs, bar_width, color = 'b', label = 'Max')
rects4 = plt.bar(index + bar_width * 3, sums, bar_width, color = 'y', label = 'sum')

plt.xticks(index, result.index.tolist())
plt.legend()
plt.show()
# %%
# 대륙별 total_litres_of_pure_alcohol을 시각화
continents = continent_mean.index.tolist()
continents.append('mean')
x_pos = np.arange(len(continents))
alcohol = continent_mean.tolist() # == continent_mean.values.tolist()
alcohol.append(total_mean)

bar_list = plt.bar(x_pos, alcohol, align='center', alpha = 0.5)
bar_list[len(continents) - 1].set_color('r')
plt.plot([0., 6], [total_mean, total_mean], 'k--')
plt.xticks(x_pos, continents)

plt.ylabel('total_litres_of_pure_alcohol')
plt.title('total_litres_of_pure_alcohol by Continent')

plt.show()
# %%
# 대륙별 beer_servings를 시각화 합니다.
beer_group = drinks.groupby('continent')['beer_servings'].sum()
continents = beer_group.index.tolist()
# print(continents)
y_pos = np.arange(len(continents))
alcohol = beer_group.tolist()

bar_list = plt.bar(y_pos, alcohol, align= 'center', alpha = 0.5)
bar_list[continents.index('EU')].set_color('r')
plt.xticks(y_pos, continents)
plt.ylabel('beer_servings')
plt.title('beer_servings by Continent')
plt.show()

# %%
# 대한민국은 얼마나 술을 독하게 마시는 나라일까
drinks['total_servings'] = drinks['beer_servings'] + drinks['wine_servings'] + drinks['spirit_servings']

# 술 소비량 대비 알코올 비율 피처
drinks['alcohol_rate'] = drinks['total_litres_of_pure_alcohol'] / drinks['total_servings']
drinks['alcohol_rate'] = drinks['alcohol_rate'].fillna(0)
# drinks['alcohol_rate'].describe

# 순의 정보 생성
country_with_rank = drinks[['country', 'alcohol_rate']]
country_with_rank = country_with_rank.sort_values(by = 'alcohol_rate', ascending= False)
country_with_rank.head(5)
# %%
# 국가별 순위 정보 시각화
country_list = country_with_rank.country.tolist()
x_pos = np.arange(len(country_list))
rank = country_with_rank.alcohol_rate.tolist()

bar_list = plt.bar(x_pos, rank)
bar_list[country_list.index('South Korea')].set_color('r')
plt.ylabel('alcohol rate')
plt.title('liquor drink rank by cotry')
plt.axis([0, 200, 0, 0.3])

korea_rank = country_list.index('South Korea')
korea_alc_rate = country_with_rank[country_with_rank['country'] == 'South Korea']['alcohol_rate'].values[0]

plt.annotate('South Korea :' + str(korea_rank + 1),
            xy = (korea_rank, korea_alc_rate),
            xytext = (korea_rank + 10, korea_alc_rate + 0.05),
            arrowprops = dict(facecolor  = 'red', shrink = 0.05))
plt.show()
# %%
