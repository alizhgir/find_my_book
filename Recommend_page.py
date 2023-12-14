import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import requests
from PIL import Image
from io import BytesIO
from IPython.core.display import HTML, display

from model.bert import recommend


list_genre = ['Классическая литература', 'Современная проза', 'Отечественные детективы',
              'Зарубежные детективы', 'Иронические детективы', 'Отечественная фантастика', 'Зарубежная фантастика',
              'Отечественное фэнтези', 'Зарубежное фэнтези', 'Ужасы', 'Фантастический боевик',
              'Российские любовные романы', 'Зарубежные любовные романы', 'Поэзия', 'Драматургия',
              'Публицистика', 'Биографии', 'Мемуары', 'Исторические романы', 'Комисксы и манга', 'Юмор',
              'Афоризмы и цитаты', 'Мифы легенды эпос', 'Сказки', 'Пословицы поговорки загадки', 'Прочие издания', 'Другое']


st.header("""
 Рекомендательная модель🤖
""", divider='blue')

st.info("""
 - ##### Именно здесь вы сможете получить ТОП-рекомендаций под ваши предпочтения и желания🔝
 \n- ##### Вам предстоит лишь сделать краткое описание книги, которую вы хотели бы прочитать, и выбрать некоторые параметры поиска⚙️
""")

st.image('images/recsys_image.png', caption='Картинка сгенерирована DALL-E')

st.write("""
  - ### Выбор параметров поиска:
""")

text_users = st.text_input('**Пожалуйста, опишите ваши предпочтения по выбору книги (какой она должна быть):**')

genre_book = st.selectbox('**Пожалуйста, укажите жанр книги:**', list_genre)

author = st.text_input('**Пожалуйста, укажите имя автора, если для вас это важно (❗НЕОБЯЗАТЕЛЬНО):**')

count_recommended = st.slider('**Пожалуйста, укажите какое количество рекомендаций Вы хотите получить:**', min_value=1, max_value=10, value=5)

push_button = st.button('**Получить рекомендации >>>**', type='primary')
start_time = time.time()

if push_button:

    recommend_book = recommend(text_users, count_recommended)

    st.write(f"""
     #### Модель нашла лучшие рекомендации для Вас🎉 :
     \n- ##### Это заняло всего {round(time.time() - start_time, 3)} сек.
    """)
    st.table(recommend_book)
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
