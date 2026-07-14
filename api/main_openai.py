import base64
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="AI Second Marker API (OpenAI Edition)")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


class TranscriptionLine(BaseModel):
    line: int
    text: str


class CVExtraction(BaseModel):
    type: str
    source_image: str
    language: str
    contains_math: bool
    transcription: List[TranscriptionLine]
    normalised_math: Dict[str, Any]
    notes: List[str]


class MarkingRequest(BaseModel):
    question: str
    mark_scheme: str
    cv_data: CVExtraction


class GradeOutput(BaseModel):
    step_by_step_analysis: str
    detected_steps: List[str]
    final_answer_detected: str
    final_answer_correct: bool
    method_marks_awarded: int
    final_marks_awarded: int
    confidence: float


def _build_grade_prompt(request: MarkingRequest) -> str:
    prompt_text = f"""
    You are an expert mathematics and calculus grader.
    You are receiving data extracted from a student's handwritten math test by a Computer Vision system.

    YOUR INSTRUCTIONS:
    1. Review the student's line-by-line transcription below.
    2. Pay close attention to the "CV Notes" section. Use these notes to determine if an error is a student's math mistake or just a messy handwriting/OCR glitch.
    3. Grade the work against the Mark Scheme.
    4. If the student makes a mistake, use the line numbers provided to identify where the logic broke down. Include this in your detected steps.

    ---EXAM DATA---
    QUESTION:
    {request.question}

    MARK SCHEME:
    {request.mark_scheme}

    ---STUDENT WORK---
    """

    for t in request.cv_data.transcription:
        prompt_text += f"[Line {t.line}]: {t.text}\n"

    prompt_text += f"""
    OCR diagnostics
    Normalised Equations Detected: {request.cv_data.normalised_math}
    CV Extraction Notes: {request.cv_data.notes}

    your task:
    You must output a strictly formatted JSON response.
    important: You MUST write a detailed explanation in the 'step_by_step_analysis' field FIRST, comparing the student's lines against the mark scheme, before you assign the final integer marks.
    """
    return prompt_text


@app.post("/mark")
async def evaluate_math(request: MarkingRequest):
    try:
        prompt_text = _build_grade_prompt(request)

        completion = client.beta.chat.completions.parse(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are an expert mathematics grader. Always return perfectly formatted data."},
                {"role": "user", "content": prompt_text},
            ],
            response_format=GradeOutput,
        )

        grade_result = completion.choices[0].message.parsed

        if grade_result is None:
            raise HTTPException(status_code=500, detail="OpenAI failed to generate a formatted response.")

        return grade_result.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(...),
    mark_scheme: str = Form(...),
):
    try:
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file was provided.")

        image_bytes = await image.read()
        content_type = image.content_type or "image/jpeg"
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        vision_prompt = """
        You are a computer vision OCR and math extraction model.
        Inspect the uploaded exam image and return valid JSON only.
        Extract handwritten text line-by-line, detect whether the page contains math, and record any OCR caveats.
        Return this exact structure:
        {
          "type": "handwritten_math_work",
          "source_image": "uploaded_image",
          "language": "en",
          "contains_math": true,
          "transcription": [{"line": 1, "text": "..."}],
          "normalised_math": {"equation_1": "..."},
          "notes": ["..."]
        }
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an OCR specialist. Return valid JSON matching the requested schema.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base64_image}",
                            },
                        },
                    ],
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content or "{}"
        extracted_data = json.loads(content)

        cv_data = CVExtraction(
            type=extracted_data.get("type", "handwritten_math_work"),
            source_image=extracted_data.get("source_image", "uploaded_image"),
            language=extracted_data.get("language", "en"),
            contains_math=bool(extracted_data.get("contains_math", True)),
            transcription=[
                TranscriptionLine(line=int(item.get("line", idx + 1)), text=str(item.get("text", "")))
                for idx, item in enumerate(extracted_data.get("transcription", []))
            ],
            normalised_math=extracted_data.get("normalised_math", {}),
            notes=list(extracted_data.get("notes", [])),
        )

        report = MarkingRequest(question=question, mark_scheme=mark_scheme, cv_data=cv_data)
        return await evaluate_math(report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))