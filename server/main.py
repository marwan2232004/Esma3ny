from fastapi import (
    FastAPI,
    File,
    UploadFile,
)
from openai import OpenAI
import dotenv
from io import BytesIO

dotenv.load_dotenv()
app = FastAPI()
client = OpenAI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/translate-any")
async def translate(txt: str, language: str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"Translate this {txt} into {language} directly without any other words",
            }
        ],
    )
    print(completion.choices[0].message.content)
    return {"response": completion.choices[0].message.content}


@app.post("/translate")
async def translate(txt: str, language: str):

    return {"response": ""}


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):

    return {"response": ""}
