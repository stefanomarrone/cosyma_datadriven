from datetime import datetime
from fastapi import APIRouter
from typing import List

router = APIRouter()

@router.get("/train")
def train(modelid: int, modelversion: int, trolleyids: List[str]) -> dict:
    #todo call the function making the training
    return {"success": True}

@router.get("/predict")
def postddmodel(modelid: int, modelversion: int, trolleyid: str, start: datetime, end: datetime) -> dict:
    predicted_rul = 0
    std = 0
    #todo call the function making the prediction
    return {"rul": predicted_rul, "standard_deviation": std}