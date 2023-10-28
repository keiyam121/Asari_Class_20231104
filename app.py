import streamlit as st
import pandas as pd
import requests
from PIL import Image
import numpy as np

targets = ['愛知県産', '千葉県産', '中国産']

st.title('アサリ分類器')
st.write('愛知県産、千葉県産、中国産のアサリを分類します')
uploaded_image=st.file_uploader("ファイルアップロード", type='jpg')





if uploaded_image is not None:
    image=Image.open(uploaded_image,)
    img_array = np.array(image)
    st.image(img_array,use_column_width = None)
   

    # 予測の実行
    if st.button("Predict", key="predict_button"):
        response = requests.post("http://localhost:8000/predict", files={'file': uploaded_image.getvalue()})
        result=response.json()
        prediction = result["prediction"]
        probability = result["probability"]
 
        # 予測結果の表示
        st.write('## Prediction')
        prob=pd.DataFrame([probability],columns=targets)
        st.write(prob)


        # 予測結果の出力
        st.write('## Result')
        st.write('このアサリはきっと',str(targets[int(prediction)]),'です!')
    
    
