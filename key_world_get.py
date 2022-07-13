'''
Date: 2022.07.08
Title: 추출: 키워드 정보 추출하기
By: Kang Jin Seong
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver
from bs4 import BeautifulSoup
import re
import time
import os

#텍스트 데이터 전처리하기
def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글의 정규표현식을 나타냅니다.
    result = hangul.sub('', text)
    return result

# 데이터 읽기
df = pd.read_csv("C:/Users/USER/Desktop/DSP_python/DataAnalysis/text_crawling.csv")

# 모든 데이터에 전처리 적용하기
df['title'] = df['title'].apply(lambda x: text_cleaning(x))
df['category'] = df['category'].apply(lambda x: text_cleaning(x))
df['content_text'] = df['content_text'].apply(lambda x: text_cleaning(x))
df.head(5)
# %%
# 말 뭉치 만들기
title_corpus = "".join(df['title'].tolist())
category_corpus = "".join(df['category'].tolist())
content_text_corpus = "".join(df['content_text'].tolist())
print(title_corpus)
# %%
