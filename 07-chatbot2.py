import streamlit as st
from langchain_openai import ChatOpenAI


# Streamlit UI 설정
st.set_page_config(page_title="ChatOpenAI Demo", page_icon=":robot:")
st.header("ChatOpenAI Demo")

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password"
    )

# 세션 상태 초기화
if (
    "messages" not in st.session_state
):  # session_state에 messages가 정의되어 있지 않다면
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }  # role에 system은 한번만 지정을 해주면 됨
    ]

# 대화 히스토리 표시
for message in st.session_state.messages:
    if (
        message["role"] != "system"
    ):  # role이 system이면 message의 content만 마크다운으로 표시해라
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 사용자 입력 처리
# prompt = st.chat_input("무엇을 도와드릴까요?")
# if prompt
# 와 같음
if prompt := st.chat_input(
    "무엇을 도와드릴까요?"
):  
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue")
        st.stop()

    # ChatOpenAI 모델 초기화
    chat = ChatOpenAI(api_key=openai_api_key, temperature=0)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in chat.stream(st.session_state.messages):
            full_response += response.content or ""
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 스크롤을 최하단으로 이동
st.empty()
