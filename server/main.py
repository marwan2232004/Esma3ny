from fastapi import (
    FastAPI,
    File,
    UploadFile,
)
from openai import OpenAI
import dotenv
from io import BytesIO

from utils import solution
import tempfile
import os

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


@app.post("/predict-audio")
def upload_audio(file: UploadFile = File(...)):
    # Read the uploaded audio file into memory
    # contents = await file.read()

    # # Create a temporary file and write the audio content to it
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
    #     tmp_file.write(contents)
    #     tmp_file_path = tmp_file.name  # Get the path of the temp file

    # try:
    #     # Pass the path of the saved file to the predict function
    #     result = predict("D:\\DEPI\\Esma3ny\\server\\Test Arabic.mp3")
    # finally:
    #     # Clean up the temporary file after prediction
    #     os.remove(tmp_file_path)
    result = solution.predict("D:\\DEPI\\Esma3ny\\server\\Test Arabic.mp3")
    return {"response": result}
