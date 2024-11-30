import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


st.title("법정감염병 Q&A 💬")


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

if "api_key" not in st.session_state:
    st.session_state['api_key']=None


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    st.session_state['api_key'] = st.text_input(
        "OPENAI API 키 입력:",
        type = 'password', # 비밀번호 입력처럼 마스킹 처리됨.
        placeholder= '여기에 API 키를 입력하세요.'
    )

    # API 키 입력 여부 확인
    if st.session_state['api_key']:
        st.success('API 키가 입력되었습니다')
    else:
        st.warning('API 키를 입력하세요.')


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


DB_path = './chroma_db'

def create_retriever():
    persist_db = Chroma(
        persist_directory= DB_path,
        embedding_function=OpenAIEmbeddings(openai_api_key =st.session_state['api_key']),
        collection_name='my_db'
        )
    retriever = persist_db.as_retriever()
    return retriever


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model=model_name, 
                     temperature=0,
                     openai_api_key = st.session_state['api_key'])

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

custom_retriever = create_retriever()

chain = create_chain(custom_retriever, model_name='gpt-4o')
st.session_state["chain"] = chain

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # API key 가 입력이 되었는지부터 먼저 확인
    if not st.session_state['api_key']:
        warning_msg.error('먼저 API 키를 입력하세요')

    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")
