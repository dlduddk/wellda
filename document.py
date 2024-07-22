import cohere
import os
from utils import *
from dotenv import load_dotenv
load_dotenv()

class Document_:
    def __init__(self):
        self.co = cohere.Client(os.environ['COHERE_API_KEY'])

    def query_refine(self, conversation, query):
        return query_refiner(conversation, query)
    
    def find_match_(self, query):
        return find_match(query)
    
    def rerank_contexts(self,query, context, n):
        if len(context)>0:
            results = self.co.rerank(model="rerank-multilingual-v2.0", query=query, documents=context, top_n=n)
            contexts = []
            for idx, r in enumerate(results):
                contexts.append(str(idx) + '. ' + r.document['text'] + '\n')
        return contexts
    
    def context_to_string(self, contexts, query):
            context = '\n'.join(contexts)
            #print(context)
            if len(context) > 2000:
                context = context[:2000]
            if (len(context + query)) > 2500:
                context = context[:2500 - len(query)]
            return context
