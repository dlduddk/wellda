
import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory


import streamlit as st
from streamlit_chat import message
from utils import *
from document import Document_
from chat import Chatbot_

from dotenv import load_dotenv
import tiktoken
import time
import pandas as pd
import re
st.set_page_config(layout="wide")
st.sidebar.success("Select a menu above.")

def encoding_getter(encoding_type: str):
    return tiktoken.encoding_for_model(encoding_type)

def tokenizer(string: str, encoding_type: str) -> list:
    encoding = encoding_getter(encoding_type)
    tokens = encoding.encode(string)
    return tokens

def token_counter(string: str, encoding_type: str) -> int:
    num_tokens = len(tokenizer(string, encoding_type))
    return num_tokens

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.1)

load_dotenv()


left, right = st.columns(spec=[0.4,0.6],gap='medium')
height = 600
 # 스크롤 제어를 위한 JavaScript 코드data-test-scroll-behavior="normal"
js = '''
    <script>
        var container = window.parent.document.querySelectorAll('div[data-testid="stVerticalBlockBorderWrapper"]')[3];
        console.log(container);
        console.log(container.scrollTop);
        container.scrollIntoView({behavior: "smooth"});
        container.scrollTop = container.scrollHeight; 
        console.log(container.scrollHeight);
        console.log(container.scrollTop);

    </script>
'''


with left.container(height=height):

    response_container = st.container()
    textcontainer = st.container()


    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["무엇을 도와드릴까요?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    if 'buffer_memory' not in st.session_state:
        st.session_state.update({"buffer_memory": ConversationBufferWindowMemory(k=1, return_messages=True), "user_id": None})

    llm = Chatbot_()
    document = Document_()
    conversation = llm.getConversation(st.session_state['buffer_memory'])

    with textcontainer:
        query = st.text_input("Query: ", key="input")

        if query:

            with st.spinner("typing..."):
                conversation_string = st.session_state['buffer_memory'].load_memory_variables({})['history']

                context = document.find_match_(document.query_refine(conversation_string,query))
                
                if context != []:
                    contexts = document.rerank_contexts(query, context, 3)
                    st.session_state.contexts = contexts
                    context = document.context_to_string(contexts,query)

                try:
                    if context:
                        response = conversation.predict(input=f"Context:{context}\nQuery:{query}\nAnswer:")
                        
                        response = "(자동화된 답변) " + llm.llm.predict(text=f"""콘텐츠 : {response}
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
                        response = f"(자동화된 답변) {llm.mention}"

                except Exception as ex:
                    response = f"(자동화된 답변)_{llm.mention}\nError: {ex}"
            
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    with response_container:
        if st.session_state['responses']:

            if len(st.session_state['responses']) >= 2:

                for i in range(len(st.session_state['responses']) - 1):
                    with st.chat_message("ai"):
                        st.write(st.session_state['responses'][i], key=str(i))

                    if i < len(st.session_state['requests']):
                        with st.chat_message("user"):
                            st.write(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
        

            with st.chat_message("ai"):
                text = st.session_state['responses'][-1]
                st.write_stream(stream_data(text))
            temp = st.empty()
            with temp:
                st.components.v1.html(js)
                print("!!!!!!!!!!!!!!!")
                time.sleep(.01) # To make sure the script can execute before being deleted
            temp.empty()  

                        
    
with right.container(height=height):
    if 'contexts' in st.session_state and st.session_state.contexts:
        st.write("검색된 문서")

        pattern = re.compile(r'(\d+)\.\s키워드\s:(.+?)\s답변\s:\s(.+)')
        extracted_data = []
        for item in st.session_state.contexts:
            match = pattern.match(item)
            
            if match:
                index, keyword, answer = match.groups()

                extracted_data.append([int(index), keyword, answer.strip()])

        df = pd.DataFrame(extracted_data, columns=['Index', '키워드', '문서'])
        df = df[['키워드', '문서']]

        #st.table(df)
        st.markdown(df.to_html(render_links=True, escape=False), unsafe_allow_html=True)

        #st.dataframe(df, width=3500)
