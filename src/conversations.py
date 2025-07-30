import os
import yaml
import streamlit as st
from openai import AsyncOpenAI
from dotenv import load_dotenv

def read_prompts():
    try:
        _ = load_dotenv()
        with open("configurations/prompts.yml", 'r') as f:
            st.session_state.prompt = yaml.safe_load(f)
    except Exception as e:
        print(f"Error occured: {str(e)}")

if not "prompt" in st.session_state:
    read_prompts()

@st.cache_resource
def get_client() -> AsyncOpenAI:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

async def generate_model_response():
    client = get_client()

    model = st.session_state.cfg_model
    system_prompt = st.session_state.prompt[model]["prompt"]

    messages = st.session_state.messages
    prompted_messages = [{ "role": "system", "content": system_prompt }] + messages
    stream = await client.chat.completions.create(
        model=model,
        messages=prompted_messages,
        stream=True,
        timeout=30.0,
    )

    async for token in stream:
        if token and token.choices and token.choices[0].delta.content:
            yield token.choices[0].delta.content
    

decoder_models = ["o3-mini-2025-01-31", "o4-mini-2025-04-16", "gpt-4o-2024-05-13"]
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
    
    with st.spinner("Ачаалж байна...", show_time=True):
        with st.chat_message("assistant"):
            response = st.write_stream(generate_model_response)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    st.session_state.generating = False
    st.rerun()
