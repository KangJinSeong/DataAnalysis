'''
Date: 2022.07.07
Title: 페이지의 URL 정보 추출하기
By: Kang Jin Seong
'''
#%%
'''웹 크롤링으로 기초 데이터 수집하기'''
# 페이지의 URL 정보 추출하기
from selenium import webdriver
from bs4 import BeautifulSoup
import re
import time

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
for page in page_urls[:3]:
  print(page)
  # pass

driver.close()
driver.quit()
#%%
'''
URL 페이지 정보를 기반으로 크롤링하기

'''

driver = webdriver.Chrome(executable_path= excutable_path)
driver.get(page_urls[0])

print('+'*100)
print(driver.title)
print(driver.current_url)
print('하나의 최근 변경된 문서 크롤링')
print('+'*100)
print("\n")
# time.sleep(2)
req = driver.page_source
soup = BeautifulSoup(req, 'html.parser')
contents_table_title = soup.find(name = 'div', attrs = {"class": "ep7T0OS0"})
title = contents_table_title.find_all('h1')[0]

contents_table_category = soup.find(name = 'div', attrs = {"class": "b1799a5a"})

category = contents_table_category.find_all('li')[0]
category_head = contents_table_category.find_all('h2')[0]

category_list = []
for category_head_parameter in category_head:
  category_list.append(category_head_parameter.text)
  category_list.append('|')
for category_parameter in category:
  category_list.append(category_parameter.text)
  category_list.append('|')
category_corpus = "".join(category_list)

content_paragraphs = soup.find_all(name = 'div', attrs ={"class": "HMXdAcwv"})
# print(content_paragraphs)

# content_paragraphs = soup.find(name = 'div', attrs ={"class": "Z7B4zv6v"})
# content_paragraphs_parameter = content_paragraphs.find_all('p')[0]
# content_paragraphs_parameter = content_paragraphs[0]
content_corpus_list = []

for paragraphs in content_paragraphs:
  content_corpus_list.append(paragraphs.text)
content_corpus = "".join(content_corpus_list)
print(content_corpus)


# for paragraphs in content_paragraphs_parameter:
#   content_corpus_list.append(paragraphs.text)
# content_corpus = "".join(content_corpus_list)

# print(str("제목: ") + title.text)
# print("\n")
# print(category_corpus)
# print("\n")
# print(str('사서: ') + content_corpus)

#크롤링에 사용한 브라우저를 종료합니다.
driver.close()
driver.quit()
#%%