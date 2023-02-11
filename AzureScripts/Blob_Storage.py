import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
from datasets import load_dataset

dataset = load_dataset("squad")

print(dataset["train"][1])



connectionString = ""


try:
    print("Azure Blob Storage v" + __version__ + " - Python quickstart sample")

    accessKey = "{AccessKey}"
    connectionString = "{connectionString}"
    # Quick start code goes here
except Exception as ex:
    print('Exception:')
    print(ex)


def CreateContainer(connectionString,containerName):
    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connectionString)


    container_client = blob_service_client.create_container(containerName)
    return blob_service_client

def DownloadDataset(dataset):
    dataset["train"].save_to_disk("./");
    dataset["validation"].save_to_disk("./");

def UploadBlobToContainer(containerName, local_file_name, blob_service_client):

    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=containerName, blob=local_file_name)

    print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

    # Upload the created file
    blob_client.upload_blob(local_file_name)


containerName="squaddatasetcontainer"

DownloadDataset(dataset)

blob_service_client = CreateContainer(connectionString, containerName)

UploadBlobToContainer(containerName, "squad-train.arrow", blob_service_client)
# UploadBlobToContainer(containerName, "squad-validation.arrow", blob_service_client)

