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
from langchain.llms import OpenAI
import streamlit as st

# Load environment variables from .env
load_dotenv()

# 사이드바에 OpenAI API 키 입력 필드 생성
with st.sidebar:
    OPENAI_API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
YOUTUBE_API_KEY = 'AIzaSyAkLGx57GK9kYt6eweigbVMgfkbYQzBHJc'

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
        maxResults=5
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

def summarize_text_with_langchain(text):
    template = '''당신은 동영상 내용을 기반으로 정보를 제공하는 동영상 요약 전문가입니다.
    해당 동영상 컨텐츠의 주제와 내용을 이해하기 쉽게 정리하고 요약해주세요. 
    또한 해당 내용이 민감한 내용을 담고 있거나, 가짜 정보일 위험이 있다면 주의 문구를 주세요. :\n\n{text}'''

    llm = ChatOpenAI(model_name="gpt-4", api_key=OPENAI_API_KEY)
    prompt_template = PromptTemplate.from_template(template=template)
    chain = prompt_template | llm
    
    summary = chain.invoke({"text": text})
    return summary.content

def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Streamlit 앱 레이아웃
st.title("YouTube 요약 챗봇")

if not YOUTUBE_API_KEY or not OPENAI_API_KEY:
    st.warning("API 키를 사이드바에 입력해주세요.")
else:
    query = st.text_input("검색어를 입력하세요")
    if st.button("검색"):
        if query:
            youtube = build_youtube_search(YOUTUBE_API_KEY)
            search_response = get_search_response(youtube=youtube, query=query, order='relevance')
            videos = get_video_info(search_response)
            videos_list = []
            
            for i, video in videos.items():
                st.write(f"### {i + 1}. {video['title']}")
                st.write(f"채널: {video['channelTitle']}")
                st.image(video['thumbnails'])
                st.write('')
                videos_list.append(f"{i + 1}. {video['title']}")

            
            # video_choice = st.number_input("동영상 번호를 선택하세요", min_value=1, max_value=5, step=1)
            video_choice = st.radio("동영상을 선택하세요", videos_list)

            if st.button("동영상 요약하기"):
                st.write('요약을 시작합니다.')
                selected_video = videos[video_choice - 1]
                video_id = selected_video["videoId"]
                url = f"https://www.youtube.com/watch?v={video_id}"

                with st.spinner("동영상 다운로드 중..."):
                    yt = YouTube(url)
                    filename = yt.streams.filter(only_audio=True).first().download()
                    renamed_file = filename.replace(".mp4", ".mp3")
                    os.rename(filename, renamed_file)

                with st.spinner("오디오 변환 중..."):
                    wav_file = convert_mp3_to_wav(renamed_file)

                with st.spinner("오디오 분할 및 변환 중..."):
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
                        transcripts.append(transcript['text'])

                    final_output = "\n".join(transcripts)

                    file_path = 'output.txt'

                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(final_output)

                with st.spinner("내용 요약 중..."):
                    text = load_text_from_file(file_path)
                    summary = summarize_text_with_langchain(text)

                st.write("요약된 내용:")
                st.write(summary)
        else:
            st.write("검색어를 입력해주세요.")


# # Streamlit 앱 레이아웃
# st.title("YouTube 요약 챗봇")

# if not YOUTUBE_API_KEY or not OPENAI_API_KEY:
#     st.warning("API 키를 사이드바에 입력해주세요.")
# else:
#     if "page" not in st.session_state:
#         st.session_state.page = 0

#     if st.session_state.page == 0:
#         query = st.text_input("검색어를 입력하세요")
#         if st.button("검색"):
#             if query:
#                 youtube = build_youtube_search(YOUTUBE_API_KEY)
#                 search_response = get_search_response(youtube=youtube, query=query, order='relevance')
                
#                 videos = get_video_info(search_response)
                
#                 video_choices = []
#                 for i, video in videos.items():
#                     st.image(video['thumbnails'], width=120)
#                     st.write(f"**제목**: {video['title']}")
#                     st.write(f"**채널**: {video['channelTitle']}")
#                     video_choices.append(f"{i}. {video['title']}")

#                 video_choice = st.radio("동영상을 선택하세요", video_choices)
                
#                 if st.button("동영상 요약하기"):
#                     selected_video_index = int(video_choice.split('.')[0])
#                     st.session_state.selected_video = videos[selected_video_index]
#                     st.session_state.page = 1

#     elif st.session_state.page == 1:
#         selected_video = st.session_state.selected_video
#         video_id = selected_video["videoId"]
#         url = f"https://www.youtube.com/watch?v={video_id}"

#         with st.spinner("동영상 다운로드 중..."):
#             yt = YouTube(url)
#             filename = yt.streams.filter(only_audio=True).first().download()
#             renamed_file = filename.replace(".mp4", ".mp3")
#             os.rename(filename, renamed_file)

#         with st.spinner("오디오 변환 중..."):
#             wav_file = convert_mp3_to_wav(renamed_file)

#         with st.spinner("오디오 분할 및 변환 중..."):
#             audio = AudioSegment.from_file(wav_file, format="wav")
#             total_length = len(audio)
#             length_per_chunk = 60 * 1000  # 60초

#             if not os.path.exists(".tmp"):
#                 os.mkdir(".tmp")

#             folder_path = os.path.join(".tmp", wav_file[:-4])
#             if not os.path.exists(folder_path):
#                 os.mkdir(folder_path)

#             chunks = []
#             for i in range(0, total_length, length_per_chunk):
#                 chunk_file_path = os.path.join(folder_path, f"{i}.wav")
#                 audio[i : i + length_per_chunk].export(chunk_file_path, format="wav")
#                 chunks.append(chunk_file_path)

#             transcripts = []
#             for i, chunk in enumerate(chunks):
#                 with open(chunk, "rb") as audio_file:
#                     transcript = client.audio.transcriptions.create(model="whisper-1", 
#                                                                             file=audio_file,
#                                                                             response_format="text")
#                 transcripts.append(transcript['text'])

#             final_output = "\n".join(transcripts)

#             file_path = 'output.txt'
#             with open(file_path, 'w', encoding='utf-8') as file:
#                 file.write(final_output)

#         with st.spinner("내용 요약 중..."):
#             text = load_text_from_file(file_path)
#             summary = summarize_text_with_langchain(text)

#         st.write("요약된 내용:")
#         st.write(summary)

#         if st.button("처음으로 돌아가기"):
#             st.session_state.page = 0