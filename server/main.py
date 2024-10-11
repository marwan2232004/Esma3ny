import sys
import io

from fastapi import (
    FastAPI,
    File,
    UploadFile,
)
from openai import OpenAI
import dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Inference import predict

import tempfile
import os
from utils.Translation import get_translate

dotenv.load_dotenv()
app = FastAPI()
client = OpenAI()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


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


class TranslationRequest(BaseModel):
    text: str


@app.post("/translate/auto")
async def translate(request: TranslationRequest):
    response = get_translate(request.text)
    return {"translation": response}


@app.post("/translate/en")
async def translate(request: TranslationRequest):
    # response = get_translate(request.text)
    return {"translation": "ليس بعد"}


@app.post("/audio2text")
async def upload_audio(file: UploadFile = File(...)):
    # Read the uploaded audio file into memory
    contents = await file.read()

    # Get the current working directory
    current_dir = os.getcwd()
    print(current_dir, flush=True)

    # Create a temporary file in the current working directory
    with tempfile.NamedTemporaryFile(
            dir=current_dir, delete=False, suffix=".wav"
    ) as tmp_file:
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name  # Get the path of the temp file

    try:
        # Pass the path of the saved file to the predict function
        print(f"Temporary file created at: {tmp_file_path}", flush=True)
        result = predict(tmp_file_path)
    finally:
        # Clean up the temporary file after prediction
        os.remove(tmp_file_path)
        print(f"Temporary file deleted: {tmp_file_path}", flush=True)

    return {"text": result}
