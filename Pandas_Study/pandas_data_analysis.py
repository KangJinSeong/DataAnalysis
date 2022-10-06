'''
Date: 2022.10.05
Title: 
By: Kang Jin Seong
'''
# import pandas as pd
# file_path = 'C:/Users/Kyngwon_SP/Desktop/Python/5674-833_4th/part3/auto-mpg.csv'
# df = pd.read_csv(file_path, header = None)
# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'accleration', 'model year', 'origin', 'name']
#
# print(df.head())
# print(type(df))

# print(df.shape)
# print(df.info())


# # 열 정보
# print(df.dtypes);print('\n')
# print(df.mpg.dtypes)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# print(df.describe());print('\n')
# print(df.describe(include = 'all'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 데이터 개수 확인
# print(df.count())
# print(type(df.count()))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 특정 열이 가지고 있는 고유값 확인
# unique_values = df['origin'].value_counts()
# print(unique_values);print('\n')

# print(type(unique_values)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd

file_path = 'C:/Users/Kyngwon_SP/Desktop/Python/5674-833_4th/part2/남북한발전전력량.xlsx'
df = pd.read_excel(file_path, engine= 'openpyxl')
df_ns = df.iloc[[0,5], 3:]
df_ns.index = ['south', 'north']
df_ns.columns = df_ns.columns.map(int)
print(df_ns.head())

df_ns.plot()
# %%
tdf_ns = df_ns.transpose()
tdf_ns.plot()
# %%
tdf_ns.plot(kind = 'bar')

# %%
tdf_ns.plot(kind = 'hist')
# %%
