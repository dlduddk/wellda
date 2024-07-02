
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

import streamlit as st
from streamlit_chat import message
from utils import *
from dotenv import load_dotenv
import time
import tiktoken

import pandas as pd
import re
st.set_page_config(layout="wide")

st.sidebar.success("Select a demo above.")


def encoding_getter(encoding_type: str):
    return tiktoken.encoding_for_model(encoding_type)

def tokenizer(string: str, encoding_type: str) -> list:
    encoding = encoding_getter(encoding_type)
    tokens = encoding.encode(string)
    return tokens

def token_counter(string: str, encoding_type: str) -> int:
    num_tokens = len(tokenizer(string, encoding_type))
    return num_tokens

load_dotenv()

import cohere
co = cohere.Client(os.environ['COHERE_API_KEY'])
st.subheader("웰다 챗봇")
st.text("") 

left, right = st.columns(spec=[0.4,0.6],gap='medium')

with left:
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["무엇을 도와드릴까요?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.1,
                    model_kwargs={
                        "frequency_penalty": 1.0,
                    })

    if 'buffer_memory' not in st.session_state:
        st.session_state.update({"buffer_memory": ConversationBufferWindowMemory(k=1, return_messages=True), "user_id": None})

    mention = " 해당 질문은 챗봇이 답변하기 어렵습니다. 코치님께 전달드려 답변드리도록 하겠습니다."

    system_msg_template = SystemMessagePromptTemplate.from_template(
        template=f"""당신은 헬스케어 상담사입니다.
                    반드시 CONTEXT의 내용을 참고하여 답변하라.
                    
                    순차적으로 생각한 뒤 답변하라.
                    절대 말을 지어내지 마시오 :
                    - CoT 방식을 이용하여 질문을 이해하라.
                    - ToT 방식을 사용하여 답하라.
                    - CONTEXT가 비어있다면, 다음과 같이 답하라 : {mention}
                    - 사용자의 질문에 CONTEXT만 이용하여 답변을 할 수 있는지 판단하라.
                        - CONTEXT만으로 답변을 생성할 수 없다면 다음과 같이 답변 : {mention}.
                        - 질문에 대한 답을 모르겠다면 다음과 같이 답변 : {mention}.
                        - CONTEXT의 키워드와 가이드만으로 답변을 할 수 있다면, 문법적으로 올바른 한국어 문장으로 답변.
                        - 답변을 할 때 코치 혹은 전문가와 상담하라는 말을 하지 않음
    """)

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation = ConversationChain(memory=st.session_state['buffer_memory'],
                                    prompt=prompt_template,
                                    llm=llm,
                                    verbose=False)


    response_container = st.container()
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")

        if query:
            with st.spinner("typing..."):
                conversation_string = st.session_state['buffer_memory'].load_memory_variables({})['history']
                refined_query = query_refiner(conversation_string, query)
                context = find_match(refined_query)
                #print(context)
                results = ""

                if context != []:
                    results = co.rerank(model="rerank-multilingual-v2.0", query=query, documents=context, top_n=3)
                    #print(results)
                    contexts = []

                    for idx, r in enumerate(results):
                        contexts.append(str(idx) + '. ' + r.document['text'] + '\n')
                    
                    context = '\n'.join(contexts)
                    if len(context) > 2000:
                        context = context[:2000]
                    if (len(context + query)) > 2500:
                        context = context[:2500 - len(query)]

                    st.session_state.contexts = contexts

                    #st.write(token_counter(context + query, "gpt-3.5-turbo"))
                try:
                    if context:
                        response = conversation.predict(input=f"Context:{context}\nQuery:{query}\nAnswer:")
                        #print(context)
                        response = "(자동화된 답변) " + llm.predict(text=f"""콘텐츠 : {response}
                                    톤 앤 매너 통일을 위해 아래 요구사항에 맞춰 콘텐츠를 수정하라.
                                    수정한 콘텐츠만 답하라. 어떻게 수정했는지는 말하지 않음.
                                    요구사항
                                    1. 전문가스럽지만, 부드럽고 친근한 말투. (예시: 끝맺음에 "~있어요", "~좋아요" 등을 적절히 섞어 사용)
                                    2. ((콘텐츠로부터 새로운 내용을 추가하지 않음.))
                                    3. ((질문으로 끝내지 않음.))
                                    4. 강한 어조로 단호하게 말하지 않음.
                                    5. 마무리 멘트 혹은 부가적인 의견을 추가하지 않음.
                                    6. 문법적으로 올바른 한국어
                                    수정된 콘텐츠 : """)

                    else:
                        response = f"(자동화된 답변) {mention}"

                except Exception as ex:
                    response = f"(자동화된 답변)_{mention}\nError: {ex}"

            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

with right:
    if 'contexts' in st.session_state and st.session_state.contexts:
        st.write("검색된 문서")

        pattern = re.compile(r'(\d+)\.\s키워드\s(.+?)\s답변\s:\s(.+)')

        extracted_data = []

        for item in contexts:
            #print(item)
            match = pattern.match(item)
            
            if match:
                index, keyword, answer = match.groups()
                print(keyword)
#                keyword=keyword.replace('#',',')
                extracted_data.append([int(index), keyword, answer.strip()])

        df = pd.DataFrame(extracted_data, columns=['Index', '키워드', '문서'])
        df = df[['키워드', '문서']]

        #st.table(df)
        st.markdown(df.to_html(render_links=True, escape=False), unsafe_allow_html=True)

        #st.dataframe(df, width=3500)