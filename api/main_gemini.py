from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import os


#loading API key
load_dotenv()
client = genai.Client()

app = FastAPI(title="AI Second Marker API (gemini)")


#the data that the API expects to receive from the camera/frontend
class OCRLine(BaseModel):
    line: int
    text: str
    confidence: float | None = None
    bounding_box: List[List[int]] | None = None


class CVExtraction(BaseModel):
    source_image: str | None = None
    language: str = "en"
    contains_math: bool = True

    transcription: List[OCRLine]

    parsed_tables: List[Dict[str, Any]] = Field(default_factory=list)

    normalised_math: Dict[str, Any] = Field(default_factory=dict)

    notes: List[str] = Field(default_factory=list)


class MarkingRequest(BaseModel):
    question: str
    mark_scheme: str
    cv_data: CVExtraction


class GradeOutput(BaseModel):
    reasoning: str

    detected_steps: List[str]

    mistakes: List[str]

    ecf_applied: bool

    final_answer_detected: str

    final_answer_correct: bool

    method_marks_awarded: int

    accuracy_marks_awarded: int

    total_marks: int

    confidence: float


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "AI Second Marker API"
    }



#marking endpoint
@app.post("/mark", response_model=GradeOutput)
async def evaluate_math(request: MarkingRequest):

    try:

        #convert ocr to readable lines

        student_work = ""

        for line in request.cv_data.transcription:

            student_work += f"Line {line.line}: {line.text}\n"

        prompt = f"""
You are an expert mathematics examiner.

You are acting as the SECOND MARKER.

The OCR may contain mistakes.

Ignore obvious OCR artefacts.

Use mathematical reasoning.

Follow the mark scheme carefully.

Apply Error Carried Forward (ECF) whenever allowed.

------------------------------------------------

QUESTION

{request.question}

------------------------------------------------

MARK SCHEME

{request.mark_scheme}

------------------------------------------------

STUDENT WORK

{student_work}

------------------------------------------------

OCR NOTES

{request.cv_data.notes}

------------------------------------------------

Return ONLY JSON.

Instructions:

1. Explain your reasoning.

2. Identify each mathematical step.

3. Identify mistakes.

4. State whether ECF was applied.

5. Detect the student's final answer.

6. Award method marks.

7. Award accuracy marks.

8. Calculate total marks.

Do not include markdown.

Do not include explanations outside JSON.
"""

        response = client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt,

            config=types.GenerateContentConfig(

                temperature=0,

                response_mime_type="application/json",

                response_schema=GradeOutput

            )

        )

        if response.text is None:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned an empty response."
            )

        result = json.loads(response.text)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )