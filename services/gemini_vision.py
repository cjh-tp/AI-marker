import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def read_math_paper(image_path: str):

    image = Path(image_path)

    if not image.exists():
        raise FileNotFoundError(image)

    prompt = """
You are an OCR system specialised in handwritten mathematics.

Your task:

1. Read ALL visible handwritten text.
2. Preserve the student's working.
3. Preserve mathematical notation.
4. Do NOT grade.
5. Do NOT correct mistakes.

Return ONLY JSON.

Format:

{
  "transcription":[
    {
      "line":1,
      "text":"..."
    }
  ]
}
"""

    uploaded = client.files.upload(file=image)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            uploaded,
            prompt
        ]
    )

    return response.text

if __name__ == "__main__":

    result = read_math_paper(
        "images/test_photo.png"
    )

    print(result)