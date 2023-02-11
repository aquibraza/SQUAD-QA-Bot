import json
import requests
from azure.cosmos import CosmosClient, PartitionKey, exceptions


class DataInsertionModel:

    def __init__(self) -> None:
        self.endPoint = "https://productdevelopmentstorage.documents.azure.com:443/"
        self.primaryKey = "nVds9dPOkPuKu8RyWqigA1DIah4SVZtl1DIM0zDuRKd95an04QC0qv9TQIgrdtgluZo7Z0HXACFQgKgOQEAx1g=="
        self.client = CosmosClient(self.endPoint, self.primaryKey)




    def SimulateLiveData(self, liveData):
        database = self.client.get_database_client("squadstorage")
        container = database.get_container_client("squadtestcontainer")
        container.upsert_item(liveData)

        container = database.get_container_client("processedcontainer")
        container.upsert_item(liveData)
