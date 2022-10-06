'''
Date: 2022.10.05
Title: 
By: Kang Jin Seong
'''

import pandas as pd
# file_path = 'C:/Users/Kyngwon_SP/Desktop/Python/5674-833_4th/part2/read_csv_sample.csv'

# df1 = pd.read_csv(file_path)
# print(df1);print('\n')

# df2 = pd.read_csv(file_path, header = None)
# print(df2);print('\n')

# df3 = pd.read_csv(file_path, index_col= None)
# print(df3);print('\n')

# df4 = pd.read_csv(file_path, index_col= 'c0')
# print(df4);print('\n')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# file_path = 'C:/Users/Kyngwon_SP/Desktop/Python/5674-833_4th/part2/남북한발전전력량.xlsx'

# df1 = pd.read_excel(file_path, engine='openpyxl')
# # print(df1)

# df2 = pd.read_excel(file_path, engine = 'openpyxl', header= None)
# print(df2)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = { 'name': [ 'jerry', 'riah', 'paul'],
        'algol':['A', 'A+', 'B'],
        'basic' : ['C', 'B', 'B+'],
        'C++': ['B+', 'C', 'C+']}

df = pd.DataFrame(data)
df.set_index('name', inplace= True)
print(df)

df.to_csv('C:/Users/Kyngwon_SP/Desktop/Python/df_sample.csv')
df.to_excel('C:/Users/Kyngwon_SP/Desktop/Python/df_sample.xlsx')

