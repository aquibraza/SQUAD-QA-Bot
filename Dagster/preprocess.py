import json
from os import close
from pathlib import Path
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader
import subprocess

class Model:

    def __init__(self) -> None:
        self.endPoint = "https://productdevelopmentstorage.documents.azure.com:443/"
        self.primaryKey = "nVds9dPOkPuKu8RyWqigA1DIah4SVZtl1DIM0zDuRKd95an04QC0qv9TQIgrdtgluZo7Z0HXACFQgKgOQEAx1g=="
        self.client = CosmosClient(self.endPoint, self.primaryKey)
        self.tokenizer = None

    def GetData(self, type):
        database = self.client.get_database_client("squadstorage")
        container = database.get_container_client(type)
        item_list = list(container.read_all_items(max_item_count=10))
        return item_list



    def add_end_idx(self, answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text'][0]
            start_idx = answer['answer_start'][0]
            end_idx = start_idx + len(gold_text)

            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx-1:end_idx-1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
            elif context[start_idx-2:end_idx-2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

        return answers, contexts

    def Tokenizer(self, train_contexts, train_questions, val_contexts, val_questions):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        train_encodings = self.tokenizer(train_contexts, train_questions, truncation=True, padding=True)
        val_encodings = self.tokenizer(val_contexts, val_questions, truncation=True, padding=True)

        return train_encodings, val_encodings


    def add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start'][0]))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

            # if start position is None, the answer passage has been truncated
            
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        return encodings

    # train_contexts, train_questions, train_answers = read_squad('squad/train-v2.0.json')
    # val_contexts, val_questions, val_answers = read_squad('squad/dev-v2.0.json')

    def ArrangeData(self, type):
        squad_dict = self.GetData(type)

        contexts = []
        questions = []
        answers = []

        for i in squad_dict:
            contexts.append(i["context"])
            questions.append(i["question"])
            answers.append(i["answers"])

        return contexts, questions, answers


    def ModelExecution(self):
        train_contexts, train_questions, train_answers = self.ArrangeData("livecheckcontainer")
        val_contexts, val_questions, val_answers = self.ArrangeData("livecheckcontainer")
        print(train_answers)

        train_answers, train_contexts = self.add_end_idx(train_answers, train_contexts)
        val_answers, val_contexts = self.add_end_idx(val_answers, val_contexts)

        train_encodings, val_encodings = self.Tokenizer(train_contexts, train_questions, val_contexts, val_questions)

        train_encodings  = self.add_token_positions(train_encodings, train_answers)
        val_encodings = self.add_token_positions(val_encodings, val_answers)

        train_dataset = SquadDataset(train_encodings)
        val_dataset = SquadDataset(val_encodings)

        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")


        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model.to(device)
        model.train()

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        optim = AdamW(model.parameters(), lr=5e-5)

        for epoch in range(2):
            print(epoch)
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optim.step()
        print("Done")
        model.eval()
        model.save_pretrained("./")
        self.tokenizer.save_pretrained("./")

        subprocess.call(["git", "add","--all"])
        subprocess.call(["git", "status"])
        subprocess.call(["git", "commit", "-m", "First version of the your-model-name model and tokenizer."])
        subprocess.call(["git", "push"])




class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# import requests
# API_URL = "https://api-inference.huggingface.co/models/Ateeb/QA"
# headers = {"Authorization": "Bearer api_DHnvjPKdjmjkmEYQubgvmIKJqWaNNYljaF"}

# def query(payload):
# 	data = json.dumps(payload)
# 	response = requests.request("POST", API_URL, headers=headers, data=data)
# 	return json.loads(response.content.decode("utf-8"))


# data = query(
#     {
#         "inputs": {
#             "question": "What is my name?",
#             "context": "My name is Clara and I live in Berkeley.",
#         }
#     }
# )
# print(data)