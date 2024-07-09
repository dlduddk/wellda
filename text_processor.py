import re
import csv
import os
from io import StringIO
from openai import OpenAI  # 1.8
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document

class TextProcessor:
    def __init__(self, api_key, index_name, model_name="upskyy/kf-deberta-multitask"):
        self.api_key = api_key #pinecone
        self.index_name = index_name #pinecone index name
        self.model = SentenceTransformer(model_name)
        self.pc = Pinecone(api_key=api_key)
        self.index = self._initialize_index()

    def _initialize_index(self):
        if self.index_name not in self.pc.list_indexes():
            self.pc.create_index(self.index_name,
                                 dimension=768,
                                 metric='cosine',
                                 spec=ServerlessSpec(
                                     cloud="aws",
                                     region="us-east-1")
                                 )
        return self.pc.Index(self.index_name)

    @staticmethod
    def preprocess_text(text):
        text = re.sub(r'\s+', '\n', text)
        return text

    @staticmethod
    def clean_str(text):
        pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
        text = re.sub(pattern=pattern, repl='', string=text)

        pattern = '<[^>]*>'  # HTML 태그 제거
        text = re.sub(pattern=pattern, repl='', string=text)

        text = re.sub(r'\s+', ' ', text)
        return text

    def process_csv(self, file_content):
        documents = []
        f = StringIO(file_content)
        data = csv.reader(f, delimiter=',')
        header = next(data)
        for row in data:
            content = "키워드 : " + row[2]
            content += "\n\n답변 : " + row[3]
            content = self.clean_str(content)
            metadata = {
                '번호': row[0],
                '분류': row[1],
                '키워드': row[2],
                '이미지': row[4],
                '변환 이미지': row[5],
                '레퍼런스': row[6],
                '제작 진행': row[7],
                'last edited by': row[8],
                'page_content':content
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def create_embeddings(self, documents):
        embeddings_list = []
        for doc in documents:
            input_em = self.model.encode(doc.page_content).tolist()
            embeddings_list.append(input_em)
        return embeddings_list

    def upsert_embeddings_to_pinecone(self, embeddings, documents):
        data = []
        for idx, (embedding, doc) in enumerate(zip(embeddings, documents)):
            data.append((str(idx), embedding, doc.metadata))
        self.index.upsert(vectors=data)

    def process_and_upsert_csv(self, data):
        documents = self.process_csv(data)
        embeddings = self.create_embeddings(documents)
        self.upsert_embeddings_to_pinecone(embeddings, documents)