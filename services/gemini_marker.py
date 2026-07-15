"""
gemini_marker.py

Uses Gemini to grade a student's mathematics work.

Input:
- Question
- Mark scheme
- CV/OCR extraction

Output:
- Structured JSON grading result
"""

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def grade_math(question, mark_scheme, cv_data, response_schema):
    """
    Uses Gemini to grade a student's work.

    Parameters
    ----------
    question : str

    mark_scheme : str

    cv_data : dict
        OCR / Vision output

    response_schema :
        GradeOutput Pydantic model

    Returns
    -------
    dict
    """

    # ---------------------------------------
    # Convert transcription into readable text
    # ---------------------------------------

    student_work = ""

    for line in cv_data["transcription"]:
        student_work += f"Line {line['line']}: {line['text']}\n"

    # ---------------------------------------
    # Prompt
    # ---------------------------------------

    prompt = f"""
You are an expert mathematics examiner.

You are acting as the SECOND MARKER.

The OCR transcription may contain small mistakes.

Ignore obvious OCR artefacts.

Follow the mark scheme carefully.

Apply Error Carried Forward (ECF) whenever appropriate.

--------------------------------------------

QUESTION

{question}

--------------------------------------------

MARK SCHEME

{mark_scheme}

--------------------------------------------

STUDENT WORK

{student_work}

--------------------------------------------

OCR NOTES

{cv_data.get("notes", [])}

--------------------------------------------

Return ONLY JSON.

You MUST include:

- reasoning
- detected_steps
- mistakes
- ecf_applied
- final_answer_detected
- final_answer_correct
- method_marks_awarded
- accuracy_marks_awarded
- total_marks
- confidence

Do not include markdown.
"""

    response = client.models.generate_content(

        model="gemini-2.5-flash",

        contents=prompt,

        config=types.GenerateContentConfig(

            temperature=0,

            response_mime_type="application/json",

            response_schema=response_schema

        )

    )

    if response.text is None:
        raise Exception("Gemini returned an empty response.")

    return json.loads(response.text)