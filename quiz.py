import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# 데이터 파일 경로
filename = "./09_irisdata/09_irisdata.csv"

# 컬럼 이름 정의
column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# 데이터 읽어오기
data = pd.read_csv(filename, names=column_names)

# 데이터 셋의 행렬 크기(shape)
print("데이터 셋의 행렬 크기:", data.shape)

# 데이터 셋의 요약(describe())
print("\n데이터 셋의 요약:")
print(data.describe())

# 데이터 셋의 클래스 종류(groupby('class').size())
print("\n데이터 셋의 클래스 종류:")
print(data.groupby('class').size())

# scatter_matrix 그래프 저장
scatter_matrix(data, diagonal='hist')
plt.savefig("scatter_matrix.png")
plt.close()

# 독립 변수 X (X는 0~3번 속성)와 종속 변수 Y(Y는 4번 속성)로 분할
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# 학습 모델은 의사결정 나무 사용(튜닝 X)
model = DecisionTreeClassifier()

# K-fold(10개의 폴드 지정), cross validation(평가 지표 accuracy)
kfold = KFold(n_splits=10, random_state=42, shuffle=True)

# K-fold의 평균 정확도 출력
results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
print("\nK-fold의 평균 정확도:", results.mean())
