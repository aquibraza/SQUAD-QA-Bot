from azure import cosmos
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from datasets import load_dataset
import time
import random
import requests
import json  

class CosmosStorage:

    def __init__(self) -> None:
        self.endPoint = "https://productdevelopmentstorage.documents.azure.com:443/"
        self.primaryKey = "nVds9dPOkPuKu8RyWqigA1DIah4SVZtl1DIM0zDuRKd95an04QC0qv9TQIgrdtgluZo7Z0HXACFQgKgOQEAx1g=="
        self.client = CosmosClient(self.endPoint, self.primaryKey)

    def CreateDatabase(self, databaseName):
        database = self.client.create_database_if_not_exists(id=databaseName)


    def CreateContainer(self, databaseName, containerName) -> None:
        container_name = containerName
        database = self.client.get_database_client(databaseName)

        container = database.create_container_if_not_exists(
            id=container_name, 
            partition_key=PartitionKey(path="/id"),
            offer_throughput=400
        )

    def InsertPresetDataIntoContainer(self, databaseName, containerName) -> None:
        dataset = load_dataset("squad")

        database = self.client.get_database_client(databaseName)
        container = database.get_container_client(containerName)

        datatype = ""
        if (containerName == "squadtraincontainer" or containerName == "processedcontainer"):
            datatype = "train"
        elif (containerName == "squadtestcontainer"):
            datatype = "validation"


        tracker = 0
        for i in dataset[datatype]:
            container.upsert_item(i)
            tracker = tracker + 1
            if (tracker % 1000 == 0):
                print(datatype + " - Batch number: " + str(tracker) + " Uploaded")

        print("Insertion Complete")
        

    def SimulateLiveData(self):
        dataset = load_dataset("squad")
        datatype = "train"


        while (True):
            time.sleep(10)

            tracker = random.randint(0, len(dataset[datatype]))
            url = 'http://127.0.0.1:8000/input'
            json_object = json.dumps(dataset[datatype][tracker])  
            print(json_object) 

            x = requests.post(url, data = json_object)

    def ReturnData(self, databaseName, containerName):
        database = self.client.get_database_client(databaseName)
        container = database.get_container_client(containerName)
        item_list = list(container.read_all_items(max_item_count=10))
        print(item_list)

        return item_list



cosmosStorage = CosmosStorage()
# cosmosStorage.CreateDatabase("squadstorage")
# cosmosStorage.CreateContainer("squadstorage","squadtraincontainer")
# cosmosStorage.CreateContainer("squadstorage","livecheckcontainer")
# cosmosStorage.InsertPresetDataIntoContainer("squadstorage","squadtraincontainer")
# cosmosStorage.InsertPresetDataIntoContainer("squadstorage","processedcontainer")
# cosmosStorage.SimulateLiveData()
# cosmosStorage.InsertPresetDataIntoContainer("squadstorage","squadtestcontainer")
# cosmosStorage.ReturnData("squadstorage","livecheckcontainer")

























