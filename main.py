from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import time
import asyncio

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
        # Create the image generation request
        async with httpx.AsyncClient() as client:
            create_response = await client.post(
                "https://api.starryai.com/api/v1/create",
                json={
                    "prompt": request.prompt,
                    "height": 512,
                    "width": 512,
                    "cfg_scale": 7.5,
                    "seed": None,
                    "samples": 1
                },
                headers={
                    "accept": "application/json",
                    "content-type": "application/json",
                    "authorization": f"Bearer {os.getenv('STARRYAI_API_KEY')}"
                }
            )
            
            if create_response.status_code != 200:
                raise HTTPException(status_code=create_response.status_code, detail=create_response.text)
            
            creation_id = create_response.json().get('id')
            
            # Poll for the result
            max_attempts = 30
            attempt = 0
            while attempt < max_attempts:
                status_response = await client.get(
                    f"https://api.starryai.com/api/v1/generations/{creation_id}",
                    headers={
                        "accept": "application/json",
                        "authorization": f"Bearer {os.getenv('STARRYAI_API_KEY')}"
                    }
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get('status') == 'succeeded':
                        return {
                            "imageUrl": status_data['images'][0]['url'],
                            "status": "success"
                        }
                    elif status_data.get('status') == 'failed':
                        raise HTTPException(status_code=500, detail="Image generation failed")
                
                attempt += 1
                await asyncio.sleep(2)  # Wait 2 seconds between checks
            
            raise HTTPException(status_code=408, detail="Image generation timed out")
                
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
