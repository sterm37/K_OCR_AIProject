import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# python = 3.7.11
#필요 라이브러리 목록

# 모델 학습 및 테스트 필요 라이브러리(기본적으로 설치하라 하셨던 것들)
# tensorflow: 2.8.0 (pip install tensorflow==2.8.0)
# numpy: 1.21.5 (conda install numpy==1.21.5)
# sklearn: 1.0.2 (pip install sklearn==1.0.2)

# 손글씨 사진을 실시간으로 받아오기 위한 WebApp을 실행하기 위한 라이브러리들
# cv2: 4.1.2 (pip install opencv-python==4.1.2)
# pandas: 1.3.5 (pip install pandas==1.3.5)
# streamlit: 1.10.0 (pip install streamlit==1.10.0)
# streamlit_drawable_canvas: 0.9.1 (pip install streamlit_drawable_canvas==0.9.1)
# matplotlib: 3.2.2 (conda install matplotlib=3.2.2)

# 손글씨를 이미지로 실시간으로 받아오기 위한 WebApp 입니다.
# 해당 코드는 가중치만 불러오는게 아니라 계층 구성도 까지 불러와야 하기에 전체적인 모델 저장 파일들이 필요하며
# 현재 가장 성능이 좋다 판단한 모델을 불러오게 설정하였습니다.

@st.cache(allow_output_mutation = True) #코드를 불러올시 딱 한번만 사용되게 하는 코드
def load_model(name): #모델을 불러오는 부분
    model = tf.keras.models.load_model('./models/'+name)
    return model

label = ['20대(여)', '30대(여)','40대(여)', '20대(남)', '30대(남)', '40대(남)']
model = load_model('vgg11')

st.write('# 내 글씨체 판별기')

col1, col2 = st.columns(2) #이미지를 나타낼 공간

with col1:# 글씨를 쓸 공간
    canvas = st_canvas(
        fill_color = '#FFFFFF',
        stroke_width = 3,
        stroke_color = '#000000',
        background_color = '#FFFFFF',
        width = 128, 
        height = 128,
        drawing_mode = 'freedraw', 
        key = 'canvas'
    )
    
if canvas.image_data is not None: #캔버스에 이미지가 있으면 실행
    # image load
    img = canvas.image_data.astype('uint8')#uint8 타입으로 이미지 받아옴
    img = cv2.resize(img,(32, 32)) #opencv를 통해 이미지 리사이징
    preview_img = cv2.resize(img, (192, 192)) #어떻게 리사이징이 되었는가 표시할 이미지
    
    col2.image(preview_img) # 리사이징 이미지 표시
    
    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #해당 이미지를 흑백으로 전환
    x = np.array(x, dtype=np.float32) # 이미지를 numpy 배열로 전환
    x = x.reshape((-1, 32, 32, 1)) # 해당 배열을 모델에 집어 넣을 수 있는 크기로 변환
    
    y = model.predict(x).squeeze() #이미지를 모델에 집어넣어 예측
    result = np.argmax(y) #argmax를 통해 가장 높은 값이 있는 위치 인덱스를 받아옴
    
    st.write(f'## Result: {label[result]}') # 예측 결과 값 반환
    
    most_arg = y.argsort()[::-1][:5] # 상위 5개 예측 결과값 반환
    most_val = [f'{y[idx]*100:.8f}' for idx in most_arg] #해당 결과값 전처리
    preds = [f'{label[idx]}' for idx in most_arg] #예측값들 전처리
    st.bar_chart(y) #예측값들 그래프로 표현
    chart_data = pd.DataFrame(
        np.array([most_val, preds]).T, columns=['Prob(%)', 'Pred'])# 예측값과 결과값들 표로 표현
    st.write(chart_data)# 웹에 표시하기