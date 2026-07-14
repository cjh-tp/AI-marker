from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from typing import List, Dict, Any
import ollama


app = FastAPI(title="AI Second Marker API (ollama)")


#the input
class TranscriptionLine(BaseModel):
    line: int
    text: str

class CVExtraction(BaseModel):
    type: str
    source_image: str
    language: str
    contains_math: bool
    transcription: List[TranscriptionLine]
    normalised_math: Dict[str, str]
    notes: List[str]

class MarkingRequest(BaseModel):
    question: str
    mark_scheme: str
    cv_data: CVExtraction

#the output
class GradeOutput(BaseModel):
    step_by_step_analysis: str #this part forces the AI to THINK.
    detected_steps: List[str]
    final_answer_detected: str
    final_answer_correct: bool
    method_marks_awarded: int
    final_marks_awarded: int
    confidence: float

@app.post("/mark")
async def evaluate_math(request: MarkingRequest):
    try:
        #the prompt is built dynamically based on the CV structure
        prompt_text = f"""
        You are a Mathematics and Logic Examiner grading a student's exam.
        You are evaluating handwritten student work that has been digitised via OCR. 

        grading rules:
        1. FORGIVE OCR NOISE: Ignore random symbols ('~!@#') or weird spacing caused by the scanner.
        2. CONTEXTUAL AUTOCORRECT: If the CV Extraction Notes flag a potential misread (e.g., 'x+3' instead of 'x-3'), use your mathematical intuition. If the student's subsequent steps only make sense if it was a minus sign, assume it was a minus sign and do not deduct marks.
        
        math and logic rules:
        3. BOOLEAN EQUIVALENCE (LOGIC): You must recognize De Morgan's laws, commutativity, and distributivity. If the mark scheme requires 'p AND (q OR r)', strictly accept '(r OR q) AND p'. 
        4. TRUTH TABLES (LOGIC): Evaluate tables column by column. If a student calculates an intermediate column incorrectly, but applies the final logical operator correctly based on their flawed intermediate column, you MUST award the final method mark (Error Carried Forward).
        5. ALGEBRAIC EQUIVALENCE (MATH): Accept unsimplified fractions, un-factored polynomials, or logically equivalent expressions UNLESS the mark scheme explicitly demands the answer in its "simplest form".
        
        error carried forward (ECF):
        6. Apply strict ECF for multi-step logic proofs and calculus problems. Only deduct marks for the specific step where the mathematical/logical error occurred, not the subsequent steps that rely on it.

        exam data:
        question: 
        {request.question}

        marking scheme:
        {request.mark_scheme}

        student work
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
        

        #calling local ollama instance
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': 'You are a strict math grading AI. Always return perfectly formatted JSON.'},
                {'role': 'user', 'content': prompt_text}
            ],
            #forces local llama to use pydantic schema
            format=GradeOutput.model_json_schema(), 
            options={'temperature': 0} #keeps the grading strict and logical
        )


        #ollama
        result_json = json.loads(response['message']['content'].strip())

        return result_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))