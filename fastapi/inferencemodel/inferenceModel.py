import json
import requests

class InferenceModel:

    def __init__(self) -> None:
        self.API_URL = "https://api-inference.huggingface.co/models/Ateeb/QA"
        self.headers = {"Authorization": "Bearer api_DHnvjPKdjmjkmEYQubgvmIKJqWaNNYljaF"}



    def query(self, payload):
        data = json.dumps(payload)
        response = requests.request("POST", self.API_URL, headers=self.headers, data=data)
        return json.loads(response.content.decode("utf-8"))

