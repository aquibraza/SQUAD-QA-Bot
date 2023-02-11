from pydantic import BaseModel
import requests

class ModelArguments(BaseModel):
    topic: str
    question:str

class LiveData(BaseModel):
    context: str
    idUnique: str
    question:str
    title: str

 
# wikipediaDataExtractor = WikipediaDataExtractor()
# wikipediaDataExtractor.ExtractText("Unix")


