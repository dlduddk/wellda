from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

from transformers import BertTokenizerFast  # !pip install transformers
from collections import Counter

from transformers import AutoModel, AutoTokenizer

load_dotenv()

client = OpenAI(api_key="sk-proj-qxd4kxEWuSUu63VJku8cT3BlbkFJBbEhfGvVbYsNnbQWdBqL")
    
model = SentenceTransformer("upskyy/kf-deberta-multitask")
pc1 = Pinecone(api_key=os.environ['pinecone_api'])
index1 = pc1.Index(host='https://wellda-1apu0lr.svc.aped-4627-b74a.pinecone.io') #https://dw-1apu0lr.svc.gcp-starter.pinecone.io / https://dw-6netppr.svc.gcp-starter.pinecone.io

def hybrid_scale(dense, sparse, alpha: float):
    # check alpha value is in range
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def find_match(input):

    input_em = model.encode(input).tolist()
    sparse_vec = generate_sparse_vectors(input)

    dense_vec, sparse_vec = hybrid_scale(
      input_em, sparse_vec, alpha=0.5
   )
    result1 = index1.query(vector=dense_vec,sparse_vector= sparse_vec, top_k=30, include_values=True,include_metadata=True,)
    r=[]
    result1.matches.sort(key=lambda x: x.score, reverse=True)
    print(result1)
    for idx, res in enumerate(result1.matches):
        if res.score >= 0.3:
          r.append(result1['matches'][idx]['metadata']['page_content'])

    return r


def query_refiner(conversation, query):

    
    response = client.completions.create(model="gpt-3.5-turbo-instruct",
                                         prompt=f"""Please clarify user's query.
                                         Query: {query}
                                         Never answer, just refine the user's query.
                                         Refined query:""",
    temperature=0.1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return response.choices[0].text



def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):

        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    client.completions.create(model="gpt-3.5-turbo-instruct",
    prompt=f"""Please Summarize the following conversation to no more than 2,000 tokens
    CONVERSATION LOG: {conversation_string}
    
    SUMMARY: """)
    return conversation_string
def build_dict(input):
    # store a batch of sparse embeddings
    sparse_emb = []
    # iterate through input batch
    for token_ids in input:
        indices = []
        values = []
        # convert the input_ids list to a dictionary of key to frequency values
        d = dict(Counter(token_ids))
        for idx in d:
            indices.append(idx)
            values.append(float(d[idx]))
        sparse_emb = {'indices': indices, 'values': values}
    # return sparse_emb list
    return sparse_emb


def generate_sparse_vectors(context_batch):
    # create batch of input_ids
    tokenizer = BertTokenizerFast.from_pretrained(
        'klue/bert-base'
    )
    inputs = tokenizer(
            context_batch, padding=True,
            truncation=True,
            max_length=512
    )['input_ids']
    # create sparse dictionaries
    sparse_embeds = build_dict([inputs])
    return sparse_embeds
