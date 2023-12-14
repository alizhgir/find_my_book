import pandas as pd
import numpy as np
import torch
import faiss

from transformers import AutoTokenizer, AutoModel


weight = "cointegrated/rubert-tiny2"

tokenizer = AutoTokenizer.from_pretrained(weight)
model = AutoModel.from_pretrained(weight)

vectors_annotation = np.load('datasets/annotation_embeddings2.npy')
data_frame = pd.read_csv('datasets/cleaned_final_books.csv')

MAX_LEN = 512

faiss_index = faiss.IndexFlatL2(312)

faiss_index.add(vectors_annotation)


def recommend(text, top_k):

    tokenized_text = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=MAX_LEN)
    tokenized_text = torch.tensor(tokenized_text).unsqueeze(0)

    with torch.inference_mode():
        predict = model(tokenized_text)

        vector = predict[0][:, 0, :].squeeze().cpu().numpy()

    vector = np.array([vector])
    value_metrics, index = faiss_index.search(vector, k=top_k)

    recommend_books = data_frame.iloc[index.reshape(top_k,)][['category_name', 'author', 'title', 'age', 'annotation']].reset_index(drop=True)
    recommend_books = recommend_books.rename({'category_name': 'Жанр', 'author': 'Автор', 'title': 'Название книги', \
                                              'age': 'Возрастное ограничение', 'annotation': 'Аннотация'}, axis=1)

    return recommend_books




