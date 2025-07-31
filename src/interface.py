import os
import yaml
import asyncio
import streamlit as st

from openai import AsyncOpenAI
from dotenv import load_dotenv
from models import ModelCapsule

@st.cache_resource
def read_capsule():
    capsule = ModelCapsule()
    return capsule

def read_prompts():
    try:
        _ = load_dotenv()
        with open("configurations/prompts.yml", 'r') as f:
            st.session_state.prompt = yaml.safe_load(f)
    except Exception as e:
        print(f"Error occured: {str(e)}")

if not "prompt" in st.session_state:
    read_prompts()

capsule = read_capsule()

decoder_models = capsule.model_labels
encoder_models = ["Jina/Jina-v3-embedding", "Qwen/Qwen3-4B-Embedding"]

@st.dialog("Промптын тохиргоо өөрчлөх", width="large")
def update_prompt():
    model = st.session_state.cfg_model
    st.write(f"{model} хэлний загварын промпт өөрчилж байна.")
    if not st.session_state.prompt[model]:
        prompt_chosen = ""
    else:
        prompt_chosen = st.session_state.prompt[model]
    prompt_updated = st.text_area("Одоогийн промпт", prompt_chosen["prompt"], height="content")
    if st.button("Хадгалах"):
        if prompt_chosen != prompt_updated:
            st.session_state.prompt[model]["prompt"] = prompt_updated
            with open(f"configurations/prompts.yml", "w") as f:
                yaml.dump(st.session_state.prompt, f)
        
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar arrangement
st.logo("resources/chimege.png", size="large")

st.sidebar.header("Тохиргоо")
if st.sidebar.button("✨ Шинэ чат үүсгэх", use_container_width=True):
    del st.session_state.messages 
    st.session_state.messages = []

st.sidebar.selectbox("Хэлний загвар сонгох (LLM)", decoder_models, key="cfg_model")
st.sidebar.slider("Температур", 0.0, 1.0, value=0.25, key="cfg_temperature")
if st.sidebar.button("Системийн промпт тохируулах", use_container_width=True):
    update_prompt()


st.sidebar.divider()

st.sidebar.header("Нэмэлт тохиргоо")
st.sidebar.checkbox("Загварын контекст протокол ашиглах (MCP)", value=True, key="cfg_tool_use")

if st.session_state.cfg_tool_use:
    st.sidebar.selectbox("Асуулт эмбедлэх загвар (SBERT)", encoder_models)
    st.sidebar.checkbox("Асуултыг шууд векторт хөрвүүлж маягт хайх", value=True)
    st.sidebar.slider("Вектор хайлтын радиус", 0.0, 1.0, value=0.25)


st.sidebar.divider()
st.sidebar.write("© 2025 Эгүнэ AI.")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Flag to prevent the if statement running again
if "generating" not in st.session_state:
    st.session_state.generating = False

if user_input := st.chat_input("Юу асуумаар байна?", disabled=st.session_state.generating):
    st.session_state.generating = True
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)
    
    current_model_id = st.session_state.cfg_model
    messages = st.session_state.messages

    with st.spinner("Ачаалж байна...", show_time=True):
        with st.chat_message("assistant"):
            if current_model_id.split('/')[0] == "google":
                print(f"GENAI")
                response = st.write_stream(capsule._generate_genai(current_model_id, messages[-1]["content"]))
            else:
                print(f"OPENAI/EGUNE")
                response = st.write_stream(capsule._generate_openai(current_model_id, messages))
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    st.session_state.generating = False
    st.rerun()
