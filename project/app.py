import streamlit as st
import bert_search
import tfidf
import bm
import time

col1, col2 = st.columns([2.5, 1])

from PIL import Image
image = Image.open('data\heart.jpg')

with col1:
    st.title("Поиск ответов о любви")

with col2:
    st.markdown("""<style> .st.Image > {{margin-left: - 5px; width: 50}}""", unsafe_allow_html= True)
    st.image(image, width = 50)

with col1:
    query = st.text_input("Введите ваш вопрос")

    search_type= st.radio("Способ поиска", ["tf-idf", "bm-25", "bert"])

    if st.button("Искать"):
        if query == '':
            st.markdown('#### Задан пустой запрос!')
        else:
            start = time.time()
            if search_type == "tf-idf":
                answers = tfidf.main(query)
            elif search_type == "bm-25":
                answers = bm.main(query)
            elif search_type == "bert":
                answers = bert_search.main(query)

            finish = time.time()
            st.write('Запрос обработан за {} секунд'.format(round(finish - start, 2)))
            with st.container():
                for answer in answers:
                    st.write(answer)


    

