import logging

from azure.core.credentials import AzureSasCredential
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

log = logging.getLogger(__name__)


class AzureTableSource:
    """Pulls all entities from a single Azure Table."""

    def __init__(self, name: str, url: str, sas_token: str) -> None:
        self.name = name
        self.url = url
        self.sas_token = sas_token

    def fetch(self) -> list:
        try:
            table_service_client = TableServiceClient(
                endpoint=self.url, credential=AzureSasCredential(self.sas_token)
            )
            table_client = table_service_client.get_table_client(table_name=self.name)
            entities = []
            for entity in table_client.list_entities():
                entity["Timestamp"] = str(entity._metadata["timestamp"])
                entities.append(entity)
            return entities
        except Exception:
            log.exception(f"Error! Check {self.name} authorization parameters")
            return []


class AzureBlobSource:
    """Pulls blob metadata for every blob in a single container."""

    def __init__(self, name: str, account_url: str, sas_token: str, container_name: str) -> None:
        self.name = name
        self.account_url = account_url
        self.sas_token = sas_token
        self.container_name = container_name

    def fetch(self) -> list:
        try:
            blob_service_client = BlobServiceClient(
                account_url=self.account_url, credential=self.sas_token
            )
            container_client = blob_service_client.get_container_client(self.container_name)
            blob_details = []
            for blob in tqdm(container_client.list_blobs()):
                blob_details.append(
                    {
                        "name": blob.name,
                        "memory_mb": float(blob.size / pow(1024, 2)),
                        "container": blob.container,
                        "creation_time_utc": blob.creation_time,
                    }
                )
            return blob_details
        except Exception:
            log.exception(f"Error! Check {self.name} authorization parameters")
            return []


def resolve_table_credentials(auth_config: dict, key_ref: str) -> tuple:
    entry = auth_config["tables"][key_ref]
    return entry["url"], entry["sas_token"]


def resolve_blob_credentials(auth_config: dict, key_ref: str) -> tuple:
    account_url = auth_config["blobs"]["account_url"]
    sas_token = auth_config["blobs"][key_ref]["sas_token"]
    return account_url, sas_token, key_ref
