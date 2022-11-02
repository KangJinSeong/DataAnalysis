'''
Date: 2022.10.31
Title: 
By: Kang Jin Seong
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train  = pd.read_csv('C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/titanic_train.csv')
df_test = pd.read_csv('C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/titanic_test.csv')
df_train.head()
# %%
print(df_train.info())
print('--------------')
print(df_test.info())
# %%
''' 불필요한 피처 제거'''
df_train = df_train.drop(['name', 'ticket', 'body', 'cabin','home.dest'], axis = 1)
df_test = df_test.drop(['name',  'ticket', 'body', 'cabin','home.dest'], axis = 1)
df_test.info()
# %%
'''탐색적 데이터 분석하기'''
print(df_train['survived'].value_counts())
df_train['survived'].value_counts().plot.bar()
# %%
# survived 피처를 기준으로 그룹을 나누어 그룹별 pclass 피처와 분포를 살펴봅니다.
print(df_train['pclass'].value_counts)
ax = sns.countplot(x = 'pclass', hue = 'survived', data = df_train)
# %%
from scipy import stats

# 두 집단의 피처를 비교해주며 탐색작업을 자동화하는 함수를 정의합니다.
def valid_features(df, col_name, distribution_check = True):
    
    # 두 집단(survived = 1, survived = 0 )의 분포 그래프를 출력합니다.
    g = sns.FacetGrid(df, col = 'survived')
    g.map(plt.hist, col_name, bins = 30)
    
    # 두 집단(survived = 1, survived = 0 )의 표준편차를 각각 출력합니다.
    titanic_survived = df[df['survived'] == 1]
    titanic_survived_static = np.array(titanic_survived[col_name])
    print('data std is', '%.2f' %np.std(titanic_survived_static))
    
    titanic_n_survived = df[df['survived'] == 0]
    titanic_n_survived_static = np.array(titanic_n_survived[col_name])
    print("data std is", '%.2f' %np.std(titanic_n_survived_static))
    
    # T-test로 두 집단의 평균 차이를 검정합니다.
    tTestResult = stats.ttest_ind(titanic_survived[col_name], titanic_n_survived[col_name])
    tTestResultDiffVar = stats.ttest_ind(titanic_survived[col_name], titanic_n_survived[col_name], equal_var = False)
    print('The t-statistic and p-value assuming equal variances is %.3f and %.3f' %tTestResult)
    print('The t-statistic and p-value not assuming equal variances is %.3f and %.3f' %tTestResultDiffVar)
    
    if distribution_check:
        # Shapiro-wilk 검정: 분포의 정규성 정도를 검증합니다.
        print('The w-statistic and p-value in Survived %.3f and %.3f' %stats.shapiro(titanic_survived[col_name]))
        print('The w-statistic and p-value in Non-Survived %.3f and %.3f' %stats.shapiro(titanic_n_survived[col_name]))


# %%
valid_features(df_train[df_train['age']>0], 'age', distribution_check= True)
valid_features(df_train, 'sibsp', distribution_check= False)
# %%
'''분류 모델을 위해 전처리 하기'''

# age의 결측값을 평균값으로 대체
replace_mean = df_train[df_train['age']>0]['age'].mean()
df_train['age'] = df_train['age'].fillna(replace_mean)
df_test['age'] = df_test['age'].fillna(replace_mean)

# embark: 2개의 결측값을 최빈값으로 대체합니다.
embarked_mode = df_train['embarked'].value_counts().index[0]
df_train['embarked'] = df_train['embarked'].fillna(embarked_mode)
df_test['embarked'] = df_test['embarked'].fillna(embarked_mode)

# 원 핫 인코딩을 위한 통합 데이터 프레임을 생성
whole_df = df_train.append(df_test)
train_idx_num = len(df_train)

# pandas 패키를 이용한 원-핫 인코딩을 수행한다.
whole_df_encoded = pd.get_dummies(whole_df)
df_train = whole_df_encoded[:train_idx_num]
df_test = whole_df_encoded[train_idx_num:]

df_train.head()
# %%
'''분류 모델링'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 학습데이터 분류
# print(df_train.columns)
x_train, y_train = df_train.loc[:, df_train.columns != 'survived'].values, df_train['survived'].values
x_test, y_test = df_test.loc[:, df_test.columns != 'survived'].values, df_test['survived'].values

# 로지스틱 회귀 모델을 학습합니다.
lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)

# 예측 결과를 반환합니다.
y_pred = lr.predict(x_test)
y_pred_probabilty = lr.predict_proba(x_test)[:,1]
# %%
'''분류 모델 평가하기'''
# Confusion Matrix 활용
print("accuracy: %.2f" %accuracy_score(y_test, y_pred))
print("Precision: %.3f" %precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))
# %%
# Confusion Matrix 직접 출력
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred = y_pred)
print(confmat)
# %%
# 분류 직전인 확률값 y_pred_probablity인 0-1 사이의 값을 사용
from sklearn.metrics import roc_curve, roc_auc_score
# AUC 출력
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_probabilty)
roc_auc = roc_auc_score(y_test, y_pred_probabilty)
print('AUC : %.3f' % roc_auc)

# ROC curve를 그래프로 출력합니다.
plt.rcParams['figure.figsize'] = [5,4]
plt.plot(false_positive_rate, true_positive_rate, label = 'ROC curve(area= %0.3f)'%roc_auc, color ='red', linewidth = 4)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of Logistic regression')
plt.legend(loc="lower right")

# %%
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
y_pred_probabilty = dtc.predict_proba(x_test)[:,1]

print("accuracy: %.2f" %accuracy_score(y_test, y_pred))
print("Precision: %.3f" %precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1: %.3f' % f1_score(y_test, y_pred))

# %%
# 학습한 모델의 AUC를 계산하여 출력합니다.
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_probabilty)
roc_auc = roc_auc_score(y_test, y_pred_probabilty)
print('AUC : %.3f' % roc_auc)

# ROC curve를 그래프로 출력합니다.
plt.rcParams['figure.figsize'] = [5,4]
plt.plot(false_positive_rate, true_positive_rate, label = 'ROC curve(area= %0.3f)'%roc_auc, color ='red', linewidth = 4)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of Logistic regression')
plt.legend(loc="lower right")
# %%
# 피처 엔지니어링

df_train  = pd.read_csv('C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/titanic_train.csv')
df_test = pd.read_csv('C:/Users/USER/Desktop/DSP_python/DataAnalysis/workspace/python-data-analysis/data/titanic_test.csv')

df_train = df_train.drop(['ticket', 'body', 'home.dest'], axis = 1)
df_test = df_test.drop(['ticket', 'body', 'home.dest'], axis = 1)

# age의 결측 값을 평균값으로 대체합니다.
replace_mean = df_train[df_train['age']>0]['age'].mean()
df_train['age'] = df_train['age'].fillna(replace_mean)
df_test['age'] = df_test['age'].fillna(replace_mean)

# embark: 2개의 결측값을 최빈 값으로 대체합니다.
embarked_mode = df_train['embarked'].value_counts().index[0]
df_train['embarked'] = df_train['embarked'].fillna(embarked_mode)
df_test['embarked'] = df_test['embarked'].fillna(embarked_mode)

# 원핫 인코딩
whole_df = df_train.append(df_test)
train_idx_num = len(df_train)
# %%
# cabin 피처 활용하기
print(whole_df['cabin'].value_counts()[:10])
# print(whole_df['cabin'].head(15))
# %%
# 결측 데이터인 경우 'X'로 대체하기
whole_df['cabin'] = whole_df['cabin'].fillna('X')

# cabin 피처와 첫 번째 알파벳을 추출하기
whole_df['cabin'] = whole_df['cabin'].apply(lambda x: x[0])

# 추출한 알파벳 중 G와 T는 수가 너무 작기 때문에 X로 대체
whole_df['cabin'] = whole_df['cabin'].replace({"G":'X', "T":"X"})

ax = sns.countplot(x = 'cabin', hue = 'survived', data = whole_df)
plt.show()

# %%
# name 피처 활용하기
name_grade = whole_df['name'].apply(lambda x: x.split(',',1)[1].split('.')[0])
name_grade = name_grade.unique().tolist()
print(name_grade)
# %%
# 호칭에 따라 사회적 지위(1910년대 기준)을 정의합니다.
grade_dict = {'A': ['Rev', 'Col', 'Major', 'Dr', 'Capt', 'Sir'], # 명예직을 나타냅니다.
              'B': ['Ms', 'Mme', 'Mrs', 'Dona'], # 여성을 나타냅니다.
              'C': ['Jonkheer', 'the Countess'], # 귀족이나 작위를 나타냅니다.
              'D': ['Mr', 'Don'], # 남성을 나타냅니다.
              'E': ['Master'], # 젊은남성을 나타냅니다.
              'F': ['Miss', 'Mlle', 'Lady']} # 젊은 여성을 나타냅니다.

# 정의한 호칭의 기준에 따라, A~F의 문자로 name 피처를 다시 정의하는 함수입니다.
def give_grade(x):
    grade = x.split(", ", 1)[1].split(".")[0]
    for key, value in grade_dict.items():
        for title in value:
            if grade == title:
                return key
    return 'G'
    
# 위의 함수를 적용하여 name 피처를 새롭게 정의합니다.
whole_df['name'] = whole_df['name'].apply(lambda x: give_grade(x))
print(whole_df['name'].value_counts())
# %%

# 원핫 인코딩
whole_df_encoded = pd.get_dummies(whole_df)
df_train = whole_df_encoded[:train_idx_num]
df_test = whole_df_encoded[train_idx_num:]
df_train.head()
# %%
# 데이터 분리
x_train, y_train = df_train.loc[:, df_train.columns != 'survived'], df_train['survived'].values
x_test, y_test = df_test.loc[:, df_test.columns != 'survived'], df_test['survived'].values

lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred_probability = lr.predict_proba(x_test)[:,1]

# 테스트 데이터셋에 대한 accuracy, precision, recall, f1 평가 지표를 각각 출력합니다.
print("accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision : %.3f" % precision_score(y_test, y_pred))
print("Recall : %.3f" % recall_score(y_test, y_pred))
print("F1 : %.3f" % f1_score(y_test, y_pred)) # AUC (Area Under the Curve) & ROC curve


# AUC(Area Under the Curve)를 계산하여 출력합니다.
false_positive_rate, true_positive_rate, thresholds  = roc_curve(y_test, y_pred_probability)
roc_auc = roc_auc_score(y_test, y_pred_probability)
print("AUC: %.3f" % roc_auc)

# ROC curve를 그래프로 출력합니다.
plt.rcParams['figure.figsize'] = [5, 4]
plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.3f)' % roc_auc, 
         color='red', linewidth=4.0)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of Logistic regression')
plt.legend(loc="lower right")
# %%

'''피처 영향력 알아보기'''
cols  = df_train.columns.tolist()
cols.remove('survived')
y_pos = np.arange(len(cols))

# 각 피처별 회귀 분석 계수를 그래프의 x축으로 하여 , 피처 영향력 그래프를 출력합니다.
plt.rcParams['figure.figsize'] = [5,4]
fig, ax = plt.subplots()
ax.barh(y_pos, lr.coef_[0], align = 'center', color = 'green', ecolor = 'balck')
ax.set_yticks(y_pos)
ax.set_yticklabels(cols)
ax.invert_yaxis()
ax.set_xlabel('Coef')
ax.set_title('Each Feature Coef')

plt.show()

# %%
