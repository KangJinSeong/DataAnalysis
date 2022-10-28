#%%
'''
Date: 2022.10.27
Title: 
By: Kang Jin Seong
'''

import time
import operator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



rating_file_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/ml-1m/ratings.dat'
movie_file_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/ml-1m/movies.dat'
user_file_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/ml-1m/users.dat'

rating_data = pd.io.parsers.read_csv(rating_file_path,
names = ['user_id', 'movie_id', 'rating', 'time'],
delimiter = "::", encoding = 'ISO-8859-1')

movie_data = pd.io.parsers.read_csv(movie_file_path,
names = ['movie_id', 'title', 'genre'],
delimiter = "::", encoding = 'ISO-8859-1')

user_data = pd.io.parsers.read_csv(user_file_path,
names = ['user_id', 'gender', 'age', 'occupation', 'zipcode'],
delimiter = "::", encoding = 'ISO-8859-1')

rating_data.head()
# %%
movie_data.head()

# %%
user_data.head()

#%%
'''정보 탐색하기'''
# 총 영화 개수
print('total number of moive in data:', len(movie_data['movie_id'].unique()))
# 연도별 영화 개수가 많은 top 10
movie_data['year'] = movie_data['title'].apply(lambda x: x[-5:-1])
movie_data['year'].value_counts().head(10)

# %%
'''장르의 속성 탐색하기'''
unique_genre_dict = {}
for index, row in movie_data.iterrows():
    genre_combination = row['genre']
    parsed_genre = genre_combination.split('|')
    for genre in parsed_genre:
        if genre in unique_genre_dict:
            unique_genre_dict[genre] += 1
        else:
            unique_genre_dict[genre] = 1
# print(unique_genre_dict)
unique_genre ={}
for i in unique_genre_dict:
    unique_genre[i] = [unique_genre_dict[i]]
# print(unique_genre)
unique_genre = pd.DataFrame(unique_genre)

plt.rcParams['figure.figsize'] = [20,16]
sns.barplot(data = unique_genre)
plt.title('Popular genre in movies' )
plt.ylabel('Count of Genre', fontsize = 12)
plt.ylabel('Genre', fontsize = 12)
plt.show()

# 유저의 정보 탐색
print('total number of user in data:', len(user_data['user_id'].unique()))

# %%
'''평점 데이터의 정보 탐색하기'''
movie_rate_count = rating_data.groupby('movie_id')['rating'].count().values

fig = plt.hist(movie_rate_count, bins = 200)
plt.ylabel('Count');plt.xlabel('Movie"s rated count')
plt.show()
print('total number of movie in data:', len(movie_data['movie_id'].unique()))
print('total number of movie rated below 100:', len(movie_rate_count[movie_rate_count < 100]))
# %%
'''영화 평균 평점 탐색'''
movie_grouped_rating_info = rating_data.groupby('movie_id')['rating'].agg(['count', 'mean'])
movie_grouped_rating_info.columns = ['rated_count', 'rating_mean']
movie_grouped_rating_info['rating_mean'].hist(bins = 150, grid = False)
# %%
import matplotlib.cm as cm
'''user-movie 형태의 표로 살펴보기'''
rating_table = rating_data[['user_id', 'movie_id', 'rating']].set_index(['user_id', 'movie_id']).unstack().fillna(0)
print(rating_table.to_numpy())
print(type(rating_table.to_numpy()))
plt.imshow(rating_table.to_numpy())
plt.grid(False)
plt.xlabel('Movie');plt.ylabel('User');plt.title('User-movie Matrix')
plt.show()
# %%
'''수학적 기법을 활용해 평점 예측하기 '''
# 데이터에 SVD 적용하기

from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# SVD 라이브러리를 사용하기 위한 학습 데이터를 생성합니다.
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(rating_data[['user_id', 'movie_id', 'rating']], reader)
train_data = data.build_full_trainset()

# SVD 모델을 학습합니다.
train_start = time.time()
model = SVD(n_factors = 8, lr_all = 0.005, reg_all = 0.02, n_epochs = 100)
model.fit(train_data)
train_end = time.time()

print('training time of model: %.2f seconds' % (train_end - train_start))
# %%

target_user_id = 4
target_user_data = rating_data[rating_data['user_id'] == target_user_id]
target_user_data.head(5)
# %%
target_user_movie_rating_dict = {}

for index, row in target_user_data.iterrows():
    movie_id = row['movie_id']
    target_user_movie_rating_dict[movie_id] = row['rating']

print(target_user_movie_rating_dict)
# %%
'''타겟 유저가 보지 않는 영화 중 예상 평점이 높은 10개 선정'''
# 타겟 유저(user_id가 4인 유저)가 보지 않는 영화 정보를 테스트 데이터로 생성합니다.
test_data = []

for index , row in movie_data.iterrows():
    moive_id = row['movie_id']
    rating = 0
    if movie_id in target_user_movie_rating_dict:
        continue
    test_data.append(target_user_id, movie_id, rating)

# 타겟 유저의 평점 점수를 예측합니다.
target_user_predictions = model.test(test_data)

# 예측된 점수 중, 타겟 유저의 영화별 점수를 target_user_movie_predict_dict로 저장합니다.
def get_user_predicted_ratings(predictions, user_id, user_history):
    target_user_movie_predict_dict = {}
    