from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank


from transformers import AutoModel, AutoTokenizer

load_dotenv()

client = OpenAI()

model = SentenceTransformer("upskyy/kf-deberta-multitask")
pc1 = Pinecone(api_key=os.environ['pinecone_api'])
index1 = pc1.Index(host='https://wellda-1apu0lr.svc.aped-4627-b74a.pinecone.io') #https://dw-1apu0lr.svc.gcp-starter.pinecone.io / https://dw-6netppr.svc.gcp-starter.pinecone.io
def find_match(input):
    
    """inputs = tokenizer(input, padding=True, truncation=True, return_tensors="pt")
    input_em, _ = model(**inputs, return_dict=False)
    """
    input_em = model.encode(input).tolist()
    result1 = index1.query(vector=input_em, top_k=30, include_values=True,include_metadata=True,)
    r=[]
    result1.matches.sort(key=lambda x: x.score, reverse=True)
    
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
