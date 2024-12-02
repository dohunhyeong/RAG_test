import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.korean import stopwords
from langchain_teddynote.community.pinecone import init_pinecone_index
from langchain_teddynote.community.pinecone import PineconeKiwiHybridRetriever
from langchain_core.prompts import PromptTemplate
import pickle
import base64


st.title("법정감염병 Q&A 💬")

openai_api_key = st.secrets["api_keys"]["openai"]
pinecone_api_key = st.secrets["api_keys"]["pinecone"]


# sparse_encoder 디코딩 부분을 try-except로 감싸서 오류 확인
try:
    sparse_encoder = pickle.loads(base64.b64decode(st.secrets["pickle_data"]["sparse_encoder"]))
    st.write("Sparse encoder loaded successfully")  # 디버깅용
except Exception as e:
    st.error(f"Error loading sparse encoder: {str(e)}")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None



# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def create_retriever():
    pinecone_params = init_pinecone_index(
    index_name="dohun1",  # Pinecone 인덱스 이름
    namespace="dohun_first_trial",  # Pinecone Namespace
    api_key=pinecone_api_key,  # Pinecone API Key
    sparse_encoder_path=sparse_encoder,  # Sparse Encoder 저장경로(save_path)
    stopwords=stopwords(),  # 불용어 사전
    tokenizer="kiwi",
    embeddings= OpenAIEmbeddings(openai_api_key=openai_api_key), # Dense Embedder
    top_k=3,  # Top-K 문서 반환 개수
    alpha=0.5,  # alpha=0.75로 설정한 경우, (0.75: Dense Embedding, 0.25: Sparse Embedding)
    )
    
    # 디버깅을 위해 파라미터 출력 (민감정보는 제외)
    debug_params = {k: v for k, v in pinecone_params.items() 
                   if k not in ['api_key', 'sparse_encoder_path']}
    st.write("Pinecone parameters:", debug_params)
    
    try:
        pinecone_retriever = PineconeKiwiHybridRetriever(**pinecone_params)
        st.write("Retriever created successfully")  # 디버깅용
        return pinecone_retriever
    except Exception as e:
        st.error(f"Error creating retriever: {str(e)}")
        raise


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Please write your answer in a markdown table format with the main points.
        Be sure to include your source in your answer.
        Answer in Korean.

        # Example Format:
        (brief summary of the answer)
        (table)
        (detailed answer to the question)
        
        **출처**
        - (URL of the source)

        # Question: 
        {question}

        # Context: 
        {context}

        # Answer:
        """
    )
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model=model_name, 
                     temperature=0,
                     openai_api_key = openai_api_key)

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
