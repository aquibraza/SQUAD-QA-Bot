import json
from pydantic.types import Json
from fastapi import Request, FastAPI, Body
from dataentities.dataEntities import ModelArguments, LiveData
from inferencemodel.inferenceModel import InferenceModel
from datainsertionmodel.dataInsertionModel import DataInsertionModel
from textextractor.textextractor import WikipediaDataExtractor

app = FastAPI()




@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/input")
def inputdata(payload: dict = Body(...)):
    # print(payload)
    dataInsertionModel = DataInsertionModel()
    data = dataInsertionModel.SimulateLiveData(payload)



@app.post("/getanswers")
def getinference(modelArguments: ModelArguments):
    wikipediaDataExtractor = WikipediaDataExtractor()
    text = wikipediaDataExtractor.ExtractText(modelArguments.topic)
    inferenceModel = InferenceModel()
    question = modelArguments.question
    data = inferenceModel.query(
        {
            "inputs": {
                "question": question,
                "context": text,
            }
        }
    )
    

    return data





