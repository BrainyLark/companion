import os
import yaml
import asyncio
import streamlit as st
import weaviate
from jinja2 import Template
from openai import AsyncOpenAI
from dotenv import load_dotenv
from models import ModelCapsule
from pdfminer.high_level import extract_text
from utils.fms import clean_and_chunk_text
from utils.dbms import WeaviateDatabaseManager

@st.cache_resource
def get_db_manager():
    db = WeaviateDatabaseManager()
    return db

db = get_db_manager()

async def insert_document_into_db(chunks):
    async with weaviate.use_async_with_local() as async_client:
        results = await db.batch_insert(async_client, chunks)
        return results
    
async def query_database(query_text: str, distance: float):
    async with weaviate.use_async_with_local() as async_client:
        return await db.search_database(async_client, query_text, 5, distance)

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
encoder_models = ["Jina/Jina-v3-embedding"]

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
st.logo("resources/icon.png", size="large")

st.sidebar.header("Тохиргоо")
if st.sidebar.button("✨ Шинэ чат үүсгэх", use_container_width=True):
    del st.session_state.messages 
    st.session_state.messages = []

st.sidebar.selectbox("Хэлний загвар сонгох (LLM)", decoder_models, key="cfg_model")
st.sidebar.slider("Температур", 0.0, 1.0, value=0.25, key="cfg_temperature")
if st.sidebar.button("Системийн промпт тохируулах", use_container_width=True):
    update_prompt()

st.sidebar.divider()

st.sidebar.checkbox("Вектор өгөгдлийн сан", value=True, key="cfg_tool_use")

if st.session_state.cfg_tool_use:
    st.sidebar.selectbox("Асуулт эмбедлэх загвар (SBERT)", encoder_models)
    st.sidebar.checkbox("Асуултаа эмбедлэх", value=True, disabled=True)
    st.sidebar.slider("Вектор хайлтын радиус", 0.0, 1.0, value=0.25, key="radius")
    uploaded_file = st.sidebar.file_uploader("PDF оруулах", type="pdf")

    if uploaded_file is not None:
        if st.sidebar.button("Боловсруулах...", use_container_width=True):
            with st.spinner("Файлыг өгөгдлийн санд оруулж байна..."):
                text = extract_text(uploaded_file)
                batch_chunks = clean_and_chunk_text(text, uploaded_file.name)
                results = asyncio.run(insert_document_into_db(batch_chunks))
                st.success(f"Вектор өгөгдлийн санд {len(results._all_responses)} документ бичигдлээ.\nХугацаа (s): {results.elapsed_seconds}")

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

    system_prompt_template = st.session_state.prompt[current_model_id]["prompt"]
    template = Template(system_prompt_template)

    search_results = []

    with st.spinner("Вектор сангаас хайж байна...", show_time=True):
        search_radius = float(st.session_state["radius"])
        search_results = asyncio.run(query_database(user_input, search_radius))
        with st.expander("Хайлтын илэрцийг харах"):
            for i, result in enumerate(search_results):
                st.write(f"Докумэнт: {i + 1}")
                st.write(f"Агуулга: {result["content"]}")
                st.write(f"Аппликейшн: {result["app_id"]}")
                st.write(f"Евклидийн зай: {result["distance"]}")
                st.write(f"\n")

    system_prompt = template.render(documents=search_results)

    with st.spinner("Хэлний загвар бодож байна...", show_time=True):
        with st.chat_message("assistant"):
            if current_model_id.split('/')[0] == "google":
                response = st.write_stream(capsule._generate_genai(current_model_id, messages[-1]["content"], system_prompt))
            else:
                messages = [{"role": "system", "content": system_prompt}] + messages
                response = st.write_stream(capsule._generate_openai(current_model_id, messages))
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    st.session_state.generating = False
    st.rerun()
