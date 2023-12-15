import pandas as pd
import numpy as np
import torch
import faiss

from transformers import AutoTokenizer, AutoModel


CHECKPOINT = "cointegrated/rubert-tiny2"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModel.from_pretrained(CHECKPOINT)

vectors_annotation = np.load('datasets/annotation_embeddings2.npy')

data_frame = pd.read_csv('datasets/final_dataset.csv')

data_frame = pd.DataFrame({
    'Cсылка на книгу': data_frame['page_url'],
    'Обложка': data_frame['image_url'],
    'Инфо': data_frame[['category_name', 'age', 'title', 'author']].agg(', '.join, axis=1),
    'Аннотация': data_frame['annotation']
})

MAX_LEN = 512

faiss_index = faiss.IndexFlatL2(312)

faiss_index.add(vectors_annotation)


def recommend(query: str, top_k: int) -> pd.DataFrame:

    tokenized_text = tokenizer.encode(query, add_special_tokens=True, truncation=True, max_length=MAX_LEN)
    tokenized_text = torch.tensor(tokenized_text).unsqueeze(0)

    with torch.inference_mode():
        predict = model(tokenized_text)

        vector = np.array([predict[0][:, 0, :].squeeze().cpu().numpy()])

    value_metrics, index = faiss_index.search(vector, k=top_k)

    value_metrics = np.round(value_metrics.reshape(top_k, ))
    recommend_books = data_frame.iloc[index.reshape(top_k, ), 1:].reset_index(drop=True)

    return recommend_books, value_metrics




