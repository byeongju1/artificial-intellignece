# 인공지능(AI)란?
인공지능(AI, Artificial Intelligence)은 인간의 학습, 추론, 문제 해결과 같은 지능적인 행동을 컴퓨터가 수행할 수 있도록 하는 기술이다. AI는 다양한 알고리즘과 모델을 활용하여 데이터를 분석하고 패턴을 학습하며, 이를 기반으로 의사 결정을 내릴 수 있다.

---

## 기존 프로그래밍과 AI 프로그래밍의 차이
기존 프로그래밍(Traditional Programming)과 AI 프로그래밍(AI Programming)은 문제 해결 방식과 규칙을 생성하는 방식에서 차이가 있다.

기존 프로그래밍은 개발자가 직접 명확한 **규칙(Rule)**을 정의하는 반면,
AI 프로그래밍은 데이터(Data)와 결과(Result)를 학습하여 스스로 규칙을 찾는 방식을 사용한다.

### 1. 기존 프로그래밍 (Traditional Programming)
🔹 동작 방식
기존 프로그래밍은 다음과 같은 흐름을 따른다.

데이터 (Data) → 규칙 (Rule) → 결과 (Result)

- 프로그래머가 규칙을 직접 설정하며, 입력값이 주어지면 코드에 정의된 규칙에 따라 결과가 결정됨.
- 명확한 알고리즘과 논리를 바탕으로 동작.
- 모든 조건과 예외 처리를 개발자가 직접 코드로 작성해야 함.

### 2. AI 프로그래밍 (AI Programming)
🔹 동작 방식
AI 프로그래밍은 기존 프로그래밍과 다르게 규칙을 명시적으로 정의하지 않고, 데이터를 학습하여 규칙을 자동 생성한다.
데이터 (Data) → 결과 (Result) → 규칙 (Rule)

- 데이터와 정답(결과)을 입력하면, AI 모델이 패턴을 학습하여 규칙(Rule)을 도출.
- 학습이 끝난 후, 새로운 입력(New Input)을 넣으면 AI가 예측된 결과를 출력.
- 직접적인 규칙을 작성할 필요가 없으며, 데이터가 많을수록 성능이 향상됨.

---

## Train 데이터와 Test 데이터를 구분하는 이유
AI 모델을 학습시킬 때, 데이터를 **Train 데이터(훈련 데이터)**와 **Test 데이터(테스트 데이터)**로 구분하는 것은 필수적인 과정이다.
그 이유는 모델이 학습한 데이터에 과적합(Overfitting)되지 않고, 새로운 데이터에 대해 일반화(Generalization)할 수 있도록 하기 위함이다.

---

## AI 프로그래밍에서 데이터 입력 방법 3가지
AI 모델을 학습시키기 위해서는 데이터를 입력해야 한다.
데이터를 입력하는 방법에는 **파일(File), 라이브러리(Library), 웹 주소(URL)**를 통한 방식이 있다.

### 1. 파일(File)로 데이터 입력
파일에서 데이터를 읽어오는 방법은 가장 일반적인 방식이며, CSV, JSON, Excel, TXT 등 다양한 형식이 존재한다.

✅CSV 파일
<details>
<summary> 코드 보기</summary>
    
```
import pandas as pd

# CSV 파일에서 데이터 읽기
df = pd.read_csv("data.csv")

# 데이터 출력
print(df.head())
```
</details>

✅ JSON 파일
<details>
<summary> 코드 보기</summary>
    
```
import json

# JSON 파일 열기
with open("data.json", "r") as f:
    data = json.load(f)

# 데이터 확인
print(data)
```
</details>

✅ Excel 파일
<details>
<summary> 코드 보기</summary>
    
```
import pandas as pd

# Excel 파일에서 데이터 읽기
df = pd.read_excel("data.xlsx")

# 데이터 출력
print(df.head())
```
</details>

### 2. 라이브러리(Library)로 데이터 입력
라이브러리를 사용하면 미리 제공되는 공개 데이터셋을 쉽게 불러올 수 있다.

✅ scikit-learn 데이터셋 사용
<details>
<summary> 코드 보기</summary>
    
```
from sklearn.datasets import load_iris

# 아이리스(붓꽃) 데이터셋 불러오기
iris = load_iris()

# 데이터 확인
print(iris.data[:5])  # 첫 5개 샘플 출력
print(iris.target[:5])  # 레이블(정답) 출력
```
</details>

✅ TensorFlow 데이터셋 사용
<details>
<summary> 코드 보기</summary>
    
```
import tensorflow as tf

# MNIST 데이터셋 로드 (손글씨 이미지)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("훈련 데이터 크기:", X_train.shape)  # (60000, 28, 28)
print("테스트 데이터 크기:", X_test.shape)  # (10000, 28, 28)
```
</details>

### 3. 웹 주소(URL)로 데이터 입력
인터넷에서 실시간 데이터를 가져오는 방법으로, 웹 크롤링(Web Crawling) 또는 API 요청을 사용한다.

✅ 웹에서 CSV 파일 다운로드 후 읽기
<details>
<summary> 코드 보기</summary>
    
```
import pandas as pd

# 웹에서 CSV 파일 가져오기
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# 데이터 출력
print(df.head())
```
</details>

✅ API를 통해 JSON 데이터 가져오기
<details>
<summary> 코드 보기</summary>


```
import requests

# API 요청 (예: 공개 날씨 API)
url = "https://api.open-meteo.com/v1/forecast?latitude=37.57&longitude=126.98&daily=temperature_2m_max&timezone=Asia/Seoul"
response = requests.get(url)
data = response.json()

# 데이터 출력
print(data)
```
</details>

상황에 따라 적절한 방법을 선택!

정리된 데이터셋 사용 → 📂 파일(File)
기존 데이터셋 활용 → 📚 라이브러리(Library)
실시간 데이터 수집 → 🌐 웹 주소(URL)

---

## 딥러닝과 머신러닝의 차이점
**딥러닝(Deep Learning)**과 **머신러닝(Machine Learning)**은 인공지능(AI)의 하위 개념이지만,
머신러닝은 사람이 특징을 정의하고 학습시키는 방식, 딥러닝은 신경망을 이용해 스스로 특징을 학습하는 방식이라는 차이가 있다.

### 1. 머신러닝(Machine Learning)
머신러닝은 데이터를 이용해 패턴을 학습하고 예측하는 알고리즘을 의미하며, 사람이 직접 특징(Feature)을 정의해야 한다.

✅ 머신러닝의 특징
- 데이터를 분석하여 패턴을 학습하고, 새로운 데이터에 대해 예측을 수행.
- 모델을 학습시키기 위해 특징(Feature)을 사람이 직접 설계해야 함.
- 선형 회귀, 의사결정나무, 랜덤 포레스트, SVM, KNN 등 다양한 알고리즘이 존재.

### 2. 딥러닝(Deep Learning)
딥러닝은 머신러닝의 하위 분야로, **인공 신경망(ANN, Artificial Neural Network)**을 사용하여 학습하는 방식이다.

✅ 딥러닝의 특징
사람이 직접 특징을 정의할 필요 없이, 모델이 스스로 학습함.
다층 신경망(Neural Network)을 사용하여 복잡한 데이터(이미지, 음성, 자연어 등)도 처리 가능.
학습을 위해 대량의 데이터와 강력한 컴퓨팅 자원(GPU, TPU)이 필요함.


| 구분 | 머신러닝 (Machine Learning) | 딥러닝 (Deep Learning) |
|------|---------------------------|---------------------------|
| **특징(Feature) 정의** | 사람이 직접 정의 | 모델이 자동 학습 |
| **주요 알고리즘** | 선형 회귀, SVM, 랜덤 포레스트 등 | 인공 신경망(ANN, CNN, RNN, Transformer) |
| **데이터 필요량** | 적은 데이터로도 학습 가능 | 대량의 데이터 필요 |
| **학습 속도** | 상대적으로 빠름 | 느림 (연산량이 많음) |
| **컴퓨팅 자원** | CPU로도 가능 | GPU/TPU 필요 |
| **적용 분야** | 구조화된 데이터 (표 형태) | 이미지, 음성, 자연어 처리 |

---

## 생성형 AI(Generative AI)란?

**생성형 AI(Generative AI)**는 기존 데이터를 학습하여 **새로운 콘텐츠(텍스트, 이미지, 오디오, 비디오 등)를 생성하는 인공지능 기술**이다.  
기존의 AI가 데이터를 분석하고 분류하는 데 초점을 맞췄다면, 생성형 AI는 **새로운 데이터를 생성**하는 데 초점을 둔다.



### 1. 생성형 AI의 특징
- 기존 데이터를 학습하여 새로운 콘텐츠를 생성함  
- 텍스트, 이미지, 오디오, 비디오 등 다양한 형식의 데이터 생성 가능  
- 딥러닝 기반의 **생성 모델(GAN, VAE, Transformer 등)**을 사용  
- 자연어 처리(NLP), 예술, 코드 생성 등 다양한 분야에서 활용  

