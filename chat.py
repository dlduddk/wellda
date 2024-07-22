from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import os
from dotenv import load_dotenv
load_dotenv()


class Chatbot_:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.1,
                    model_kwargs={
                        "frequency_penalty": 1.0,
                    })
        self.mention = " 해당 질문은 챗봇이 답변하기 어렵습니다. 코치님께 전달드려 답변드리도록 하겠습니다."

        system_msg_template = SystemMessagePromptTemplate.from_template(
        template=f"""당신은 헬스케어 상담사입니다.
                    반드시 CONTEXT의 내용을 참고하여 답변하라.
                    
                    순차적으로 생각한 뒤 답변하라.
                    절대 말을 지어내지 마시오 :
                        - CoT 방식을 이용하여 질문을 이해하라.
                        - ToT 방식을 사용하여 답하라.
                        - CONTEXT가 비어있다면, 다음과 같이 답하라 : {self.mention}
                        -  사용자의 질문에 CONTEXT만 이용하여 답변을 할 수 있는지 판단하라.
                            - 사용자의 개인 건강 정보에 대한 질문에 답변을 제공하지 마라.
                              개인 건강 정보에 대한 질문(예시: 사용자의 혈당 변동성의 원인에 대한 질문, 아픈 이유에 대한 질문)에는 다음과 같이 답변 : {self.mention}
                            - CONTEXT만으로 답변을 생성할 수 없다면 다음과 같이 답변 : {self.mention}
                            - 질문에 대한 답을 모르겠다면 다음과 같이 답변 : {self.mention}
                            - CONTEXT의 키워드와 가이드만으로 답변을 할 수 있다면, 문법적으로 올바른 한국어 문장으로 답변
                            - 답변을 할 때 코치 혹은 전문가와 상담하라는 말을 하지 않음.
        """)

        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

        self.prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    def getConversation(self,memory):
        return ConversationChain(memory=memory,
                                    prompt=self.prompt_template,
                                    llm=self.llm,
                                    verbose=False)

    def predict(self):
        pass