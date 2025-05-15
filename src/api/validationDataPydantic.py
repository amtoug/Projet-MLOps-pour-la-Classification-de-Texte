from typing import List
from pydantic import BaseModel

# Ton modèle de données
class TextData(BaseModel):
    text: str
    label: int

class DataBatch(BaseModel):
    data: List[TextData]