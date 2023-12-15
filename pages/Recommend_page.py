import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import requests
from PIL import Image
from io import BytesIO

from model.bert import recommend


LIST_GENRE = ['Классическая литература', 'Современная проза', 'Отечественные детективы',
              'Зарубежные детективы', 'Иронические детективы', 'Отечественная фантастика', 'Зарубежная фантастика',
              'Отечественное фэнтези', 'Зарубежное фэнтези', 'Ужасы', 'Фантастический боевик',
              'Российские любовные романы', 'Зарубежные любовные романы', 'Поэзия', 'Драматургия',
              'Публицистика', 'Биографии', 'Мемуары', 'Исторические романы', 'Комисксы и манга', 'Юмор',
              'Афоризмы и цитаты', 'Мифы легенды эпос', 'Сказки', 'Пословицы поговорки загадки', 'Прочие издания', 'Другое']


st.header("""
 Рекомендательная модель🤖
""", divider='blue')

st.info("""
  ##### Чуть ниже Вы можете сделать краткое описание книги, которую Вы хотели бы прочитать, и выбрать некоторые параметры поиска⚙️
""")

st.image('images/recsys_image.png', caption='Картинка сгенерирована DALL-E')

st.write("""
  - ### Выбор параметров поиска:
""")

text_users = st.text_input('**Пожалуйста, опишите ваши предпочтения по выбору книги (какой она должна быть):**')

genre_book = st.selectbox('**Пожалуйста, укажите жанр книги:**', options=LIST_GENRE, index=None)

count_recommended = st.slider('**Пожалуйста, укажите какое количество рекомендаций Вы хотите получить:**', min_value=1, max_value=10, value=5)

push_button = st.button('**Получить рекомендации >>>**', type='primary')
start_time = time.time()

if push_button:

    recommend_book, value_metrics = recommend(text_users, count_recommended)

    st.write("""
     #### Модель нашла лучшие рекомендации для Вас🎉 :
    """)
    st.info(f"""
    - ##### Это заняло всего {round(time.time() - start_time, 3)} сек.
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write('##### Обложка')

    with col2:
        st.write('##### Инфо')

    with col3:
        st.write('##### Аннотация')

    with col4:
        st.write('##### Величина сходства (Евклидово расстояние)')
    st.divider()

    for index in range(count_recommended):
        col1, col2, col3, col4 = st.columns(4)

        response = requests.get(recommend_book.loc[index, 'Обложка'])

        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)

        with col1:
            st.image(image)

        with col2:
            st.write(f"{recommend_book.loc[index, 'Инфо']}")

        with col3:
            st.write(f"{recommend_book.loc[index, 'Аннотация']}")

        with col4:
            st.write(f'{value_metrics[index]}')
        st.divider()

    time.sleep(3)
    with st.sidebar:
        st.info("""
         #### Понравились ли Вам наши рекомендации? 
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.button('**Да, очень**🔥', type='primary')
        with col2:
            st.button('**Нет,можно лучше**👎🏻', type='primary')
