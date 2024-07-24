from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.tools import argparser
from dotenv import load_dotenv
from openai import OpenAI
from pytubefix import YouTube
import os
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import deepl
import streamlit as st


# Streamlit 앱 레이아웃
st.set_page_config(page_title="Youtube Video Summarizer", page_icon=":robot:")
st.header("Youtube Video Summarizer")

# 사이드바에 OpenAI API 키 입력 필드 생성
st.sidebar.title("API Key Settings")
with st.sidebar:
    OPENAI_API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
YOUTUBE_API_KEY = 'AIzaSyAkLGx57GK9kYt6eweigbVMgfkbYQzBHJc'
DEEPL_API_KEY = '6dd9714e-b419-4502-bd67-e746db6765d2:fx'

# 언어 선택
st.sidebar.title("Language Selection")
LANGUAGE_OPTIONS = {
    "English": "EN-US",
    "Korean": "KO",
    "Spanish": "ES",
    "French": "FR",
    "German": "DE",
    "Japanese": "JA",
    "Chinese": "ZH"
}
selected_language = st.sidebar.selectbox("Select Language", list(LANGUAGE_OPTIONS.keys()))

client = OpenAI(api_key = OPENAI_API_KEY)

## 유튜브 검색
def build_youtube_search(developer_key):
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=developer_key)

def get_search_response(youtube, query, order):
    search_response = youtube.search().list(
        q=query,
        order=order,
        part="snippet",
        type="video",
        maxResults=10
    ).execute()
    return search_response

def get_video_info(search_response):
    result_json = {}
    idx = 0
    for item in search_response['items']:
        if item['id']['kind'] == 'youtube#video':
            result_json[idx] = info_to_dict(
                item['id']['videoId'],
                item['snippet']['title'],
                item['snippet']['description'],
                item['snippet']['thumbnails']['medium']['url'],
                item['snippet']['channelTitle']
            )
            idx += 1
    return result_json

def info_to_dict(videoId, title, description, url, channelTitle):
    result = {
        "videoId": videoId,
        "title": title,
        "description": description,
        "thumbnails": url,
        "channelTitle": channelTitle
    }
    return result

def convert_mp3_to_wav(filepath):
    wav_file_path = filepath.replace(".mp3", ".wav")
    audio_clip = AudioFileClip(filepath)
    audio_clip.write_audiofile(wav_file_path, fps=44100, nbytes=2, codec="pcm_s16le")
    return wav_file_path

def summarize_text_with_langchain(text, language):
    template = '''당신은 동영상 내용을 기반으로 정보를 제공하는 동영상 요약 전문가입니다.
    다음 동영상 컨텐츠의 주제와 내용을 이해하기 쉽도록 소제목을 사용해 정리하고 요약해주세요. : \n\n{text}'''

    llm = ChatOpenAI(model_name="gpt-4o", api_key=OPENAI_API_KEY)
    prompt_template = PromptTemplate.from_template(template=template)
    chain = prompt_template | llm
    summary = chain.invoke({"text": text})
    summary_text = summary.content
    
    if language != "Korean":  # Translate if the selected language is not English
        translator = deepl.Translator(auth_key=DEEPL_API_KEY)
        summary_text = translator.translate_text(summary_text, 
                                           target_lang=LANGUAGE_OPTIONS[language])
    return summary_text

def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# 유튜브 동영상 검색 및 번역 시작
if not YOUTUBE_API_KEY or not OPENAI_API_KEY:
    st.warning("Please enter API keys in the sidebar.")
else:
    if "page" not in st.session_state:
        st.session_state.page = 0

    if st.session_state.page == 0:
        query = st.text_input("Enter search query")
        search_button = st.button("Search")
        if search_button:
            if query:
                youtube = build_youtube_search(YOUTUBE_API_KEY)
                search_response = get_search_response(youtube=youtube, query=query, order='relevance')
                
                st.write(selected_language)

                videos = get_video_info(search_response)
                
                video_choices = []
                idx = 0
                for i, video in videos.items():
                    idx += 1
                    st.write(f"### {i + 1}. {video['title']}")
                    st.write(f"채널: {video['channelTitle']}")
                    st.image(video['thumbnails'])
                    st.write('')
                    video_choices.append(f"{i + 1}. {video['title']}")
                    if idx == 5:
                        break

                st.session_state.video_choices = video_choices
                st.session_state.videos = videos
                
                video_choices = st.session_state.video_choices
                video_choice = st.radio("Select a video", video_choices)
                selected_video_index = int(video_choice.split('.')[0])
                st.session_state.selected_video = videos[selected_video_index]
                
            st.session_state.page = 1
            if st.button("Summarize Video"):
                st.experimental_rerun()

    elif st.session_state.page == 1:
        selected_video = st.session_state.selected_video
        video_id = selected_video["videoId"]
        url = f"https://www.youtube.com/watch?v={video_id}"

        with st.spinner("Downloading video..."):
            yt = YouTube(url)
            filename = yt.streams.filter(only_audio=True).first().download()
            renamed_file = filename.replace(".mp4", ".mp3")
            os.rename(filename, renamed_file)

        with st.spinner("Converting audio..."):
            wav_file = convert_mp3_to_wav(renamed_file)

        with st.spinner("Splitting and converting audio..."):
            audio = AudioSegment.from_file(wav_file, format="wav")
            total_length = len(audio)
            length_per_chunk = 60 * 1000  # 60초

            if not os.path.exists(".tmp"):
                os.mkdir(".tmp")

            folder_path = os.path.join(".tmp", wav_file[:-4])
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            chunks = []
            for i in range(0, total_length, length_per_chunk):
                chunk_file_path = os.path.join(folder_path, f"{i}.wav")
                audio[i : i + length_per_chunk].export(chunk_file_path, format="wav")
                chunks.append(chunk_file_path)

            transcripts = []
            for i, chunk in enumerate(chunks):
                with open(chunk, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(model="whisper-1", 
                                                                    file=audio_file,
                                                                    response_format="text")
                transcripts.append(transcript)

            final_output = "\n".join(transcripts)

            file_path = 'output.txt'
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(final_output)

        with st.spinner("Summarizing content..."):
            text = load_text_from_file(file_path)
            summary = summarize_text_with_langchain(text, selected_language)

        title = st.session_state.selected_video['title']
        channelTitle = st.session_state.selected_video['channelTitle']
        st.write(f'Title: {title}')
        st.write(f'Channel: {channelTitle}')
        st.write('Content:')
        st.markdown(summary)

        st.session_state.page = 0
        if st.button("Back to Home"):
            st.experimental_rerun()
