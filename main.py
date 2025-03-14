from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class Message(BaseModel):
    content: str

@app.post("/api/chat")
async def chat_with_gpt(message: Message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful nutrition and healthy food assistant. Provide concise, practical advice about healthy eating and cooking."
                },
                {
                    "role": "user",
                    "content": message.content
                }
            ],
            max_tokens=150
        )
        
        return {
            "reply": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
