import json
import logging
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from azure.core.credentials import AzureSasCredential
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".arw", ".jpg"}


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


class NfsFilesystemSource:
    """Walks a mounted directory tree (e.g. the NFS longterm_storage mount) and
    lists every raw/processed image file under it. Read-only: only stats and
    reads directory entries, never writes/moves/deletes anything under root."""

    def __init__(self, name: str, root: str) -> None:
        self.name = name
        self.root = Path(root)

    def fetch(self) -> list:
        if not self.root.is_dir():
            log.warning(f"{self.name}: root {self.root} is not a directory, skipping")
            return []

        entries = []
        for path in tqdm(self.root.rglob("*"), desc=self.name):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stat = path.stat()
            entries.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(self.root)),
                    "size_bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
        return entries


class GlobusEndpointSource:
    """Recursively lists a path on a Globus endpoint via the `globus` CLI
    (assumes an already-authenticated `globus` session - this repo doesn't
    manage Globus auth). Mirrors the crawl pattern in
    ~/agir-pipeline/scripts/globus_index.py, simplified for a single-writer
    SQLite target instead of that script's multiprocessing/Postgres setup."""

    def __init__(self, name: str, endpoint_id: str, root_path: str, max_workers: int = 8) -> None:
        self.name = name
        self.endpoint_id = endpoint_id
        self.root_path = root_path.rstrip("/")
        self.max_workers = max_workers

    def _list_dir(self, path: str) -> list:
        target = f"{self.endpoint_id}:{path}"
        result = subprocess.run(
            ["globus", "ls", "--format", "json", target],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.error(f"{self.name}: globus ls failed for {target}: {result.stderr[:300]}")
            return []
        return json.loads(result.stdout).get("DATA", [])

    def fetch(self) -> list:
        entries = []
        dir_queue = deque([self.root_path])
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while dir_queue:
                batch = [dir_queue.popleft() for _ in range(min(self.max_workers, len(dir_queue)))]
                for current_path, items in zip(batch, executor.map(self._list_dir, batch)):
                    for item in items:
                        full_path = f"{current_path}/{item['name']}"
                        if item.get("type") == "dir":
                            dir_queue.append(full_path)
                            continue
                        entries.append(
                            {
                                "path": full_path,
                                "relative_path": full_path[len(self.root_path):].lstrip("/"),
                                "size_bytes": item.get("size"),
                                "mtime_utc": item.get("last_modified"),
                            }
                        )
        log.info(f"{self.name}: found {len(entries)} files under {self.endpoint_id}:{self.root_path}")
        return entries


def download_blob_to_file(account_url: str, sas_token: str, container_name: str, blob_name: str, dest_path) -> None:
    """Downloads a single blob to dest_path, creating parent directories as
    needed. Only ever writes to dest_path - the source blob is untouched."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container_client = blob_service_client.get_container_client(container_name)
    with open(dest_path, "wb") as f:
        container_client.download_blob(blob_name).readinto(f)


def resolve_table_credentials(auth_config: dict, key_ref: str) -> tuple:
    entry = auth_config["tables"][key_ref]
    return entry["url"], entry["sas_token"]


def resolve_blob_credentials(auth_config: dict, key_ref: str) -> tuple:
    account_url = auth_config["blobs"]["account_url"]
    sas_token = auth_config["blobs"][key_ref]["sas_token"]
    return account_url, sas_token, key_ref
