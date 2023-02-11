import csv
from datetime import datetime, time

from dagster import daily_schedule, pipeline, repository, solid
from dagster.utils import file_relative_path
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from preprocess import Model, SquadDataset
from transformers import DistilBertForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
import subprocess
import os


@solid
def FetchData(context):
    # print('\n1.3 - Retreiving all items in a container\n')
    # database = client.get_database_client("squadstorage")
    # container = database.get_container_client("livecheckcontainer")

    # # NOTE: Use MaxItemCount on Options to control how many items come back per trip to the server
    # #       Important to handle throttles whenever you are doing operations such as this that might
    # #       result in a 429 (throttled request)
    # item_list = list(container.read_all_items(max_item_count=10))
    # context.log.info(item_list)
    data = Model()
    container = "livecheckcontainer"
    train_contexts, train_questions, train_answers = data.ArrangeData(container)
    val_contexts, val_questions, val_answers = data.ArrangeData(container)


    return train_contexts, train_questions, train_answers, val_contexts, val_questions, val_answers, data

@solid
def PreProcessModel(context, item_list):
    context.log.info("Preprocessing Model")
    train_contexts, train_questions, train_answers, val_contexts, val_questions, val_answers, data = item_list

    train_answers, train_contexts = data.add_end_idx(train_answers, train_contexts)
    val_answers, val_contexts = data.add_end_idx(val_answers, val_contexts)

    train_encodings, val_encodings = data.Tokenizer(train_contexts, train_questions, val_contexts, val_questions)

    train_encodings  = data.add_token_positions(train_encodings, train_answers)
    val_encodings = data.add_token_positions(val_encodings, val_answers)

    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    return train_dataset, val_dataset, data


@solid
def TrainModel(context, item_list):
    context.log.info("Train Model")
    train_dataset, val_dataset, data = item_list
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
    return model, data


@solid
def ValidateModel(context, item_list):
    context.log.info("Validating Model")
    return item_list

@solid
def UpdateModel(context, item_list):
    model, data = item_list
    model.save_pretrained("../Model_Experimentations/QA")
    data.tokenizer.save_pretrained("../Model_Experimentations/QA")
    os.chdir("../Model_Experimentations/QA")
    subprocess.call(["git", "add","--all"])
    subprocess.call(["git", "status"])
    subprocess.call(["git", "commit", "-m", "Updated version of the your-model-name model and tokenizer."])
    subprocess.call(["git", "push"])
    context.log.info("Updating Model")


@pipeline
def DSPipeline():
    data = Model()
    container = "livecheckcontainer"
    UpdateModel(ValidateModel(TrainModel(PreProcessModel(FetchData()))))


@daily_schedule(
    pipeline_name="DSPipeline",
    start_date=datetime(2020, 6, 1),
    execution_time=time(6, 45),
    execution_timezone="US/Central",
)

def DailyUpdationSchedule(date):
    return {
        "solids": {
            "hello_cereal": {
                "inputs": {"date": {"value": date.strftime("%Y-%m-%d")}}
            }
        }
    }

@repository
def hello_cereal_repository():
    return [DSPipeline, DailyUpdationSchedule]