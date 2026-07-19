from pathlib import Path
import json
from google import genai
from google.genai import types

# -----------------------------
# Gemini Client
# -----------------------------

client = genai.Client()

MODEL = "gemini-2.5-flash"

# -----------------------------
# Vision Function
# -----------------------------

def analyse_page(image_path: str):

    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(image_file)

    uploaded_file = client.files.upload(
        file=image_file
    )

    prompt = """
You are an AI computer vision system for mathematics assessments.

Your job is NOT to grade.

Your ONLY job is to understand the page.

Read EVERYTHING visible.

Identify:

- Question number
- Question text
- Student working
- Final answer
- Tables
- Mathematical equations
- Diagrams (if any)

Return ONLY JSON.

Schema:

{
    "question_detected": "",
    "student_working": [],
    "final_answer": "",
    "equations": [],
    "tables": [],
    "notes": []
}
"""

    response = client.models.generate_content(

        model=MODEL,

        contents=[
            uploaded_file,
            prompt
        ],

        config=types.GenerateContentConfig(

            temperature=0,

            response_mime_type="application/json"

        )

    )

    if response.text is None:
        raise ValueError("Gemini returned an empty response.")

    return json.loads(response.text)

# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    result = analyse_page("../images/current_page.png")

    print(json.dumps(result, indent=4))