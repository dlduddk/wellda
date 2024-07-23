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

1. 사용자의 질문을 이해하기 위해 CoT 방식을 이용하라.
2. 답변을 작성할 때 ToT 방식을 사용하라.
3. CONTEXT가 비어있다면, 다음과 같이 답하라: {self.mention}
4. 사용자의 질문에 CONTEXT만 이용하여 답변을 할 수 있는지 판단하라.
5. 사용자의 개인 건강 정보에 대한 질문에 답변을 제공하지 마라. 개인 건강 정보에 대한 질문(예시: 혈압 상승, 하락의 이유에 대한 질문, 소화가 잘 안되는 이유에 대한 질문)에는 다음과 같이 답변: {self.mention}
6. CONTEXT만으로 답변을 생성할 수 없다면 다음과 같이 답변: {self.mention}
7. 질문에 대한 답을 모르겠다면 다음과 같이 답변: {self.mention}
8. CONTEXT의 키워드와 가이드만으로 답변을 할 수 있다면, 문법적으로 올바른 한국어 문장으로 답변하라.
9. 절대 CONTEXT에 없는 정보를 추가하지 마라.
10. 전문가와 상담하라는 말을 하지 마라.
11. 답변 작성 후 문장 단위로 CONTEXT와 일치하는지 다시 한 번 검토하라.
12. 각 답변을 작성할 때마다 위의 지침을 체크리스트로 활용하여 모든 조건을 만족하는지 확인하라.
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
