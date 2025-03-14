from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client without proxies
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class ChatMessage(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(message: ChatMessage):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful meal AI assistant. You must respond ONLY with a JSON object containing recipe instructions. The JSON must include: id (kebab-case), name (title case), calories (number), image (Unsplash URL), and content (array of strings with numbered steps and ingredients). Do not include any text outside the JSON response."
                },
                {
                    "role": "user", 
                    "content": message.message
                }
            ],
            temperature=0.7
        )
        
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class ImageGenRequest(BaseModel):
    prompt: str

@app.post("/api/generate-image")
async def generate_image(request: ImageGenRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.starryai.com/creations/",
                json={
                    "model": "lyra",
                    "aspectRatio": "square",
                    "highResolution": False,
                    "images": 1,
                    "steps": 20,
                    "initialImageMode": "color",
                    "prompt": request.prompt
                },
                headers={
                    "accept": "application/json",
                    "content-type": "application/json",
                    "authorization": f"Bearer {os.getenv('STARRYAI_API_KEY')}"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
            return response.json()
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
