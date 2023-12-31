import streamlit as st
from st_pages import Page, show_pages

show_pages(
    [
        Page("main.py", "Home page"),
        Page('pages/Recommend_page.py', 'Recommend page'),
        Page('pages/Results.py', 'Results page')
    ]
)

st.header("""
 Проект по рекомендациям книг различного жанра📚
""", divider='blue')
st.info("### Только на этом сервисе ты сможешь найти лучший аналог своей любимой книги🔝")

st.image('images/preview_image.png', caption='Картинка сгенерирована DALL-E')

st.write("""
 ### Уникальный состав команды:
 \n- ##### Алиса Жгир
 \n- ##### Тигран Арутюнян
 \n- ##### Руслан Волощенко
""")

st.info("""
  ### Цель проекта:
  \n- ##### Построить алгоритм RecSys, способный предлагать пользователю лучшие рекомендации, \
  отталкиваясь от его предпочтений, желаний и настроения.
""")

st.info("""
 ### Задачи:
 \n- ##### Построить алгоритм парсинга информации с книжного сайта
 \n- ##### Полученные данные очистить и сделать рабочий Dataset
 \n- ##### Создать RecSys, способную делать релеватные рекомендации для конкретного пользователя
 \n- ##### Построить Streamlit приложение для общедоступного пользования
""")

st.info("""
 ### Используемые технологии (Стек проекта):
 \n- ##### Python
 \n- ##### Языковая модель ruBERT-tiny
 \n- ##### Библиотеки: BeautifulSoup4, Sentence Transformers, faiss, transformers, genim и др.
 \n- ##### Cosine Similarity, Euclidean Distance, Inner Product - как величины расстояния в процессе тестирования моделей
 \n- ##### Euclidean Distance для формирования рекомендаций
 \n- ##### Hugging Face & Streamlit
""")