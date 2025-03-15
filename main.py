from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, base64
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

async def generate_food_image(prompt: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_image(
            prompt=f"Professional food photography of {prompt}, healthy meal, overhead view on a restaurant table, high quality, 4k"
        )
        
        if response.image:
            return response.image.url
        raise HTTPException(status_code=500, detail="No image generated")
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(message: ChatMessage):
    try:
        # Get recipe from ChatGPT
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful meal AI assistant. You must respond ONLY with a JSON object containing recipe instructions. The JSON must include: id (kebab-case), name (title case), calories (number), image (leave empty string), and content (array of strings with numbered steps and ingredients). Do not include any text outside the JSON response."
                },
                {
                    "role": "user", 
                    "content": message.message
                }
            ],
            temperature=0.7
        )
        
        recipe_json = completion.choices[0].message.content
        
        # Generate image with Gemini
        image_url = await generate_food_image(message.message)
        
        # Insert generated image URL into recipe JSON
        recipe_json = recipe_json.replace('"image": ""', f'"image": "{image_url}"')
        
        return {"response": recipe_json}

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
