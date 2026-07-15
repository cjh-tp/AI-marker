from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from services.gemini_marker import grade_math
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

        result = grade_math(
            question=request.question,
            mark_scheme=request.mark_scheme,
            cv_data=request.cv_data.model_dump(),
            response_schema=GradeOutput
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )