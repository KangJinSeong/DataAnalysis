'''
Date: 2022.10.19
Title: 
By: Kang Jin Seong
'''
#%%
import pandas as pd

file_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/chipotle.tsv'

chipo = pd.read_csv(file_path, sep = '\t')
# print(chipo.shape)
# print(chipo.info())

'''Data shape'''
#  #   Column              Non-Null Count  Dtype
# ---  ------              --------------  -----
#  0   order_id            4622 non-null   int64    주문번호
#  1   quantity            4622 non-null   int64    아이템의 주문 수량
#  2   item_name           4622 non-null   object   주문한 아이템의 이름
#  3   choice_description  3376 non-null   object   주문한 아이템의 상세 선택 옵션
#  4   item_price          4622 non-null   object   주문 아이템의 가격 정보
#%%
print(chipo.head(10));print('\n')
print(chipo.columns)
print(chipo.index)
# %%
#  수치형 피처는 quantity 만 유일 따라서 order id는 문자형으로 변환
chipo['order_id']  = chipo['order_id'].astype('str')
print(chipo.describe())
# %%
# 범주형 피처 특징
print('unique():', chipo['order_id'].unique())
print('value_counts():',chipo['order_id'].value_counts())
print(len(chipo['order_id'].unique()))
print(len(chipo['item_name'].unique()))
# %%
item_count = chipo['item_name'].value_counts()
# print(item_count.iteritems())
# for i in item_count.iteritems():
#     print(i)
# 함수 사용: enumerate, .iteritems
for idx, (val, cnt) in enumerate(item_count.iteritems(), start = 1):
    print('Top', idx, ':',val,cnt)
# %%
# 아이템별 주문 개수
order_count = chipo.groupby('item_name')['order_id'].count()
print(order_count[:10])
# %%
# 아이템별 주문 총량
item_quantity = chipo.groupby('item_name')['quantity'].sum()
print(item_quantity[:10])
# %%
# 시각화
import numpy as np
import matplotlib.pyplot as plt

item_name_list = item_quantity.index.tolist()
x_pos = np.arange(len(item_name_list))
order_cnt = item_quantity.values.tolist()
# max_index = order_cnt.index(max(order_cnt))
# print('list max:', max_index)
# print(np.max(order_cnt))

# print(item_quantity['Veggie Soft Tacos'])


plt.bar(x_pos, order_cnt, align = 'center')
plt.ylabel('ordered_item_count')
plt.title('Distribution of all orderd item')
plt.tight_layout()
# %%
# 데이터 전처리: 전처리 함수 사용하기
print(chipo.info());print('\n')
chipo['item_price'].head()
# %%
chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]))
chipo.describe()
# %%
# 탐색적 분석: 스무고개로 개념적 탐색 분석하기
# 주문당 평균 계산 금액
# print(chipo.head(3))
chipo.groupby('order_id')['item_price'].sum().mean()
# %%
# 한 주문에 10달러 이상 지불한 주문 번호(id) 출력하기
chipo_orderid_group = chipo.groupby('order_id').sum()
results = chipo_orderid_group[chipo_orderid_group.item_price >= 10]
print(results[:10])
print(results.index.values)
# %%
# 각 아이템의 가격 구하기
chipo_one_item = chipo[chipo.quantity == 1]
price_per_item = chipo_one_item.groupby('item_name').min()
price_per_item.sort_values(by = 'item_price', ascending= False)[:10]
# %%
# 아이템 가격 분포 그래프를 출력합니다.
item_name_list = price_per_item.index.tolist()
x_pos  = np.arange(len(item_name_list))
item_price = price_per_item['item_price'].tolist()
plt.bar(x_pos,item_price, align = 'center')
plt.ylabel('item price($)')
plt.title('Distribution of all orderd item')
plt.tight_layout()
plt.show()

plt.hist(item_price)
plt.ylabel('counts')
plt.title('Histrogram of item price')
plt.tight_layout()
plt.show()
# %%
# 가장 비싼 주문에서 아이템이 총 몇 개 팔렸는지 구하기
chipo.groupby('order_id').sum().sort_values(by = 'item_price', ascending= False)[:5]

# %%
# 특정 아이템이 몇번 주문 되었는지 구하기
chipo_salad = chipo[chipo['item_name'] == 'Veggie Salad Bowl']
# 한 주문 내에서 중복 집계된 item_name을 제거합니다.
chipo_salad = chipo_salad.drop_duplicates(['item_name', 'order_id'])
print(len(chipo_salad))
chipo_salad.head(5)
# %%
# 특정 주문을 2개 이상 주문한 주문 횟수 구하기
chipo_chicken = chipo[chipo['item_name'] == 'Chicken Bowl']
chipo_chicken_ordersum = chipo_chicken.groupby('order_id').sum()['quantity']
# print(chipo_chicken_ordersum.head())
chipo_chicken_result = chipo_chicken_ordersum[chipo_chicken_ordersum >= 2]
print(len(chipo_chicken_result))
print(chipo_chicken_result.head())
# %%
