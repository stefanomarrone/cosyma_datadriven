import datetime
from typing import List
from pydantic import BaseModel
from dataclasses import dataclass

class TrainingRequest(BaseModel):
    modelid: int
    modelversion: int
    trolleyids : List[str]
    start: datetime.datetime
    end: datetime.datetime
