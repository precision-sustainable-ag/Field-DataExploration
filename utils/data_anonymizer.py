import base64
import os

import yaml
from cryptography.fernet import Fernet


class DataAnonymizer:
    """
    A class for anonymizing data using encryption, specifically designed
    for handling and managing encryption keys and encrypted data within
    applications.

    Attributes:
        keypath (str): Path to the encryption key used by Fernet for encryption/decryption operations.
        fernet (Fernet): A cryptography.fernet.Fernet instance initialized with the provided key.

    Methods:
        encrypt(data): Encrypts the given string data and returns a base64-encoded string.
        decrypt(encoded_encrypted_data): Decrypts a base64-encoded encrypted string.
        append_key_to_yaml(file_path, key_name="ENCRYPTION_KEY"): Appends or updates an encryption key in a YAML file.
        load_key_from_file(file_path): Loads an encryption key from a specified YAML file.
        set_key(key): Sets the encryption key for the DataAnonymizer instance.
        get_key(): Retrieves the current encryption key.
    """

    def __init__(self, keypath: str) -> None:
        """
        Initializes the DataAnonymizer with a key file path argument.

        Parameters:
            keypath (str): The path to the encryption key to use.
        """
        assert os.path.exists(keypath), "Path to encrpytion key does not exist."

        self.keypath = keypath
        self.key = self.load_key_from_file()
        # Ensure key is bytes, even if passed as a string
        self.key = self.key.encode() if isinstance(self.key, str) else self.key
        self.fernet = Fernet(self.key)

    def encrypt(self, data: str) -> str:
        """
        Encrypts the given data and returns a base64-encoded string.

        Parameters:
            data (str): The plaintext data to encrypt.

        Returns:
            str: The encrypted data, encoded as a base64 string.
        """

        encrypted_data = self.fernet.encrypt(data.encode())
        # Encode the bytes to base64 string for serialization
        encoded_encrypted_data = base64.urlsafe_b64encode(encrypted_data).decode()
        return encoded_encrypted_data

    def decrypt(self, encoded_encrypted_data: str) -> str:
        """
        Decrypts the given base64-encoded encrypted data.

        Parameters:
            encoded_encrypted_data (str): The base64-encoded encrypted data to decrypt.

        Returns:
            str: The decrypted plaintext data.
        """
        # Decode the base64 string back to bytes
        encrypted_data = base64.urlsafe_b64decode(encoded_encrypted_data.encode())
        decrypted_data = self.fernet.decrypt(encrypted_data).decode()
        return decrypted_data

    def append_key_to_yaml(
        self, file_path: str, key_name: str = "ENCRYPTION_KEY"
    ) -> None:
        """
        DEPRECATED
        ONLY USED IN THE INITIAL GENERATION OF THE KEY
        Appends or updates an encryption key in a YAML file.

        Parameters:
            file_path (str): The path to the YAML file.
            key_name (str, optional): The name of the key in the YAML document. Defaults to "ENCRYPTION_KEY".
        """
        # Check if the YAML file already exists
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                try:
                    data = yaml.safe_load(file) or {}
                except yaml.YAMLError as exc:
                    print(f"Error loading YAML file: {exc}")
                    return
        else:
            data = {}

        # Update the key in the data dictionary
        data[key_name] = self.key.decode()

        # Write the updated data back to the YAML file
        with open(file_path, "w") as file:
            yaml.safe_dump(data, file)

    def load_key_from_file(self) -> bytes:
        """
        Loads the encryption key from a file.

        Returns:
            bytes: The loaded encryption key.
        """
        with open(self.keypath, "r") as file:
            data = yaml.safe_load(file) or {}
        key = data.get(
            "ENCRYPTION_KEY", ""
        ).encode()  # Ensure key is returned as encoded bytes
        return key

    def set_key(self, key: bytes | str) -> None:
        """
        Sets the encryption key for the DataAnonymizer instance.

        Parameters:
            key (bytes or str): The new encryption key to use.
        """
        self.key = key if isinstance(key, bytes) else key.encode()
        self.fernet = Fernet(self.key)

    def get_key(self) -> str:
        """
        Returns the current encryption key as a string.

        Returns:
            str: The current encryption key.
        """
        return self.key.decode()
