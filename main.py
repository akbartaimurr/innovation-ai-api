from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

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

# Add list of curated food images
FOOD_IMAGES = [
    "https://images.unsplash.com/photo-1546069901-ba9599a7e63c", # Healthy food bowl
    "https://images.unsplash.com/photo-1555939594-58d7cb561ad1", # Grilled food
    "https://images.unsplash.com/photo-1504674900247-0877df9cc836", # Chicken dish
    "https://images.unsplash.com/photo-1512621776951-a57141f2eefd", # Vegetable dish
    "https://images.unsplash.com/photo-1473093295043-cdd812d0e601", # Pasta dish
    "https://images.unsplash.com/photo-1565299507177-b0ac66763828", # Fish dish
    "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe", # Salad bowl
    "https://images.unsplash.com/photo-1547592180-85f173990554"  # Soup dish
]

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
                    "content": f"You are a helpful meal AI assistant. You must respond ONLY with a JSON object containing recipe instructions. The JSON must include: id (kebab-case), name (title case), calories (number), image (use one of these Unsplash URLs randomly: {', '.join(FOOD_IMAGES)}), and content (array of strings with numbered steps and ingredients). Do not include any text outside the JSON response."
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

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
