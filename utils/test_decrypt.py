import json
from pathlib import Path

from utils.data_anonymizer import DataAnonymizer

# Specify the path to the encryption key file
keypath = "../keys/authorized_keys.yaml"

# Initialize the DataAnonymizer with the specified key file
anonymizer = DataAnonymizer(keypath)


def decrypt_usernames_in_json_files(folder_path: str) -> None:
    """
    Decrypts the 'Username' fields in all JSON files within a specified folder.

    This function reads each JSON file in the given folder, decrypts the 'Username' value using
    an instance of DataAnonymizer, and prints the decrypted username. This can be modified to
    update the file or perform further actions as needed.

    Parameters:
        folder_path (str): The path to the folder containing JSON files to decrypt.

    Returns:
        None
    """
    # Glob for JSON files in the specified folder
    json_files = Path(folder_path).glob("*.json")

    for file_path in json_files:
        # Open and read the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

        # Check if 'Username' exists and is encrypted
        if "Username" in data and data["Username"]:
            # Decrypt the 'Username' value
            decrypted_username = anonymizer.decrypt(data["Username"])
            print(f"Decrypted Username in {Path(file_path).name}: {decrypted_username}")
            # If you need to update the file or perform further actions, do so here


decrypt_usernames_in_json_files("../tempdata/new_json_data")
