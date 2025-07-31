import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

from typing import Dict, Any, List
from google import genai

load_dotenv()

class ModelCapsule:
    
    def __init__(self) -> None:
        self.metadata: Dict[str, Dict[str, Any]] = {
            "google/gemini-2.5-flash" : {
                "model_class": "genai",
                "model_label": "gemini-2.5-flash",
                "api_key": os.getenv("GEMINI_API_KEY"),
                "timeout": 60
            },
            "google/gemini-2.5-flash-lite" : {
                "model_class": "genai",
                "model_label": "gemini-2.5-flash-lite",
                "api_key": os.getenv("GEMINI_API_KEY"),
                "timeout": 30
            },
            "google/gemini-2.5-pro" : {
                "model_class": "genai",
                "model_label": "gemini-2.5-pro",
                "api_key": os.getenv("GEMINI_API_KEY"),
                "timeout": 120
            },
            "chimege/chat-egune-v0.5" : {
                "model_class": "chimege",
                "model_label": "egune",
                "api_key": os.getenv("EGUNE_API_KEY"),
                "base_url": os.getenv("EGUNE_BASE_URL"),
                "timeout": 90
            },
            "openai/gpt-4o-latest": {
                "model_class": "openai",
                "model_label": "chatgpt-4o-latest",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "timeout": 120
            },
            "openai/o1-mini" : {
                "model_class": "openai",
                "model_label": "o1-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "timeout": 120
            },
            "openai/o3-mini" : {
                "model_class": "openai",
                "model_label": "o3-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "timeout": 120
            }
        }

        self.genai_client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))        
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.egune_client = AsyncOpenAI(base_url=os.getenv("EGUNE_BASE_URL"), 
                                        api_key=os.getenv("EGUNE_API_KEY"))
        
        self.genai_chats = {}
        

    @property
    def model_labels(self) -> List[str]:
        return self.metadata.keys()
    
    async def _generate_genai(self, model_id: str, user_message: str):
        try:
            if model_id not in self.genai_chats:
                chat = self.genai_client.chats.create(model=self.metadata[model_id]["model_label"])
                self.genai_chats[model_id] = chat

            response = self.genai_chats[model_id].send_message_stream(user_message)

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            print(f"Error generating tokens (GenAI): {e}")
            yield f"Error: {e}"

    async def _generate_openai(self, model_id: str, messages: List[Dict[str, Any]]):
        if self.metadata[model_id]["model_class"] == "chimege":
            tmp_client = self.egune_client
        
        else:
            tmp_client = self.openai_client

        try:
            stream = await tmp_client.chat.completions.create(
                model=self.metadata[model_id]["model_label"],
                messages=messages,
                stream=True,
                timeout=self.metadata[model_id]["timeout"]
            )

            async for chunk in stream:
                if chunk and chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            print(f"Error generating tokens (OpenAI): {e}")
            yield f"Error: {e}"