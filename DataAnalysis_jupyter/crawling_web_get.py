'''
Date: 2022.07.08
Title: 크롤링: 웹 데이터 가져오기
By: Kang Jin Seong
'''

#%%
''' Beautifulsoup을 이용해 웹 크롤링하기'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver
from bs4 import BeautifulSoup
import re
import time
import os

# 윈도우용 크롬 웹드라이버 실행 결로
excutable_path = 'C:/Users/USER/Desktop/DSP_python/DataAnalysis/chromedriver.exe'
#크롤링항 사이트 주소를 정의합니다.
source_url = "https://namu.wiki/RecentChanges"

driver = webdriver.Chrome(executable_path= excutable_path)
driver.get(source_url)
print('+'*100)
print(driver.title)
print(driver.current_url)
print('나무위키 크롤링')
print('+'*100)

time.sleep(2)
#나무위키 페이지 진입해서 프로필 테이블 추출
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
contents_table = soup.find(name = "table")
table_body = contents_table.find(name="tbody")
table_rows = table_body.find_all(name = "tr")

# a 태그의href 속성을 리스트로 추출하여 크롤링할 페이지 리스트를 생성합니다.
page_url_base = "https://namu.wiki"
page_urls = []

for index in range(0, len(table_rows)):
  first_td = table_rows[index].find_all('td')[0]
  td_url = first_td.find_all('a')
  if len(td_url) > 0:
    page_url = page_url_base + td_url[0].get('href')
    if "png" not in page_url:
      page_urls.append(page_url)

# 중복 url을 제거합니다.
page_urls = list(set(page_urls))
# driver.close()
driver.quit()

#%%
''' 나무위키의 최근 변경 데이터 크롤링하기'''
#크롤링한 데이터를 프레임으로 만들기 위해 준비 합니다.
columns = ['title', 'category', 'content_text']
df = pd.DataFrame(columns = columns)
count = 0
    # 각 페이지별 '제목', '카테고리', '본문' 정보를 데이터 프레임으로 만듭니다.
try:
    for page_url in page_urls:
        count += 1 
        #사이트의 html 구조에 기반하여 크롤링을 수행합니다.
        driver = webdriver.Chrome(executable_path= excutable_path)
        driver.get(page_url)
        time.sleep(0.1)
        req = driver.page_source
        soup = BeautifulSoup(req, 'html.parser')
        contents_table_title = soup.find(name = 'div', attrs = {"class": "ep7T0OS0"})

        if len(contents_table_title.find_all('h1')) > 0:
            title = contents_table_title.find_all('h1')[0]
        else:
            title = None

        contents_table_category = soup.find(name = 'div', attrs = {"class": "b1799a5a"})

        
        # 카테고리 정보가 없는 경우를 확인합니다.
        if len(contents_table_category.find_all('li')) > 0:
            category = contents_table_category.find_all('li')[0]
        else:
            category = None

        content_paragraphs = soup.find_all(name = 'div', attrs ={"class": "HMXdAcwv"})
        content_corpus_list = []

        #페이지 내 제목 정보에서 개행 문자를 제거한 뒤 추출합니다.
        # 만약 없는 경우, 빈 문자열로 대체합니다.
        if title is not None:
            row_title = title.text.replace("\n", "")
        else:
            row_title = ""
        print(row_title)
        print(count)

        # 페이지 내 본문 정보에서 개행 문자를 제거한 뒤 추출 합니다.
        # 만약 없는 경우 , 빈 문자열로 대체합니다.
        if content_paragraphs is not None:
            for paragraphs in content_paragraphs:
                if paragraphs is not None:
                    content_corpus_list.append(paragraphs.text.replace("\n", ""))
                else:
                    content_corpus_list.append("")
        else:
            content_corpus_list.append("")
        
        # 페이지내 카테고리 정보에서 "분류"라는 단어와 개행 문자를 제거한 뒤 추출합니다.
        # 만약 없는 경우, 빈 문자열로 대체합니다.
        if category is not None:
            row_category = category.text.replace("\n", "")
        else:
            row_category = ""

        # 모든 정보를 하나의 데이터 프레임에 저장합니다.
        row = [row_title, row_category, "".join(content_corpus_list)]
        series = pd.Series(row, index = df.columns)
        df = df.append(series, ignore_index= True)
        
        # driver.close()
        driver.quit()
except:
    if not os.path.exists("C:/Users/USER/Desktop/DSP_python/DataAnalysis/text_crawling.csv"):
        df.to_csv("C:/Users/USER/Desktop/DSP_python/DataAnalysis/text_crawling.csv", mode = 'w', header= True, index= False, encoding='utf-8-sig')
    else:
         df.to_csv("C:/Users/USER/Desktop/DSP_python/DataAnalysis/text_crawling.csv", mode = 'a', header= False, index= False, encoding='utf-8-sig')
    driver.quit()


# %%
import pandas as pd
datasheet = pd.read_csv("C:/Users/USER/Desktop/DSP_python/DataAnalysis/text_crawling.csv")
datasheet
# %%
