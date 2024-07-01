import re
from openai import OpenAI #1.8
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
import csv
import os

# Initialize embedding model
model = SentenceTransformer("upskyy/kf-deberta-multitask")
# Initialize Pinecone
pc = Pinecone(api_key=os.environ['pinecone_api'])
# Define the index name
index_name = "wellda"

# Create the index if it doesn't existS
if index_name not in pc.list_indexes():
    pc.create_index(index_name, 
                          dimension=768,
                          metric='cosine',
                          spec=ServerlessSpec(
                              cloud="aws",
                              region="us-east-1") 
                              ) # 768
# Instantiate the index
index = pc.Index(index_name)


# Define a function to preprocess text
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', '\n', text)
    return text

def process_csv(file_path):
    # load csv data
    f = open(file_path, 'r',encoding='utf-8')
    data = csv.reader(f,delimiter = ',')
    header = next(data)
    # print(header)
    # 번호 / 분류 / 키워드 / 답변 / 이미지/ 변환 이미지 / 레퍼런스 / 제작 진행 / last edited by
    documents = []
    for row in data:
        content = "키워드"+row[2].strip().replace('#', '\n')
        content += "\n\n답변 : "+row[3]
        print(content)
        #documents.append(Document(page_content=content))
        metadata={'번호':row[0],
                  '분류':row[1],
                  '키워드':row[2],
                  '이미지':row[4],
                  '변환 이미지':row[5],
                  '레퍼런스':row[6],
                  '제작 진행':row[7],
                  'last edited by':row[8]}
        documents.append(Document(page_content=content, metadata=metadata))
    f.close()

    return documents

# Define a function to create embeddings
def create_embeddings(ts):
    embeddings_list = []
    for text in ts:
        input_em = model.encode(text.page_content).tolist()
        embeddings_list.append(input_em)

    return embeddings_list

# Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings, ids):
    data = []
    for idx, embedding, doc in zip(ids, embeddings, documents):
        data.append((str(idx),embedding,doc.metadata))

    index.upsert(vectors=data)

# Process a csv and create embeddings
file_path = "./data/답변가이드_240614.csv"  # Replace with your actual file path
documents = process_csv(file_path)


import re


def clean_str(text):
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    
    text = re.sub(r'\s+', ' ', text)
    return text 

for doc in documents:
    text = doc.page_content
    text=clean_str(text)
    doc.metadata['page_content'] = text

embeddings = create_embeddings(documents)
upsert_embeddings_to_pinecone(index, embeddings, [str(i) for i in range(len(embeddings))])

