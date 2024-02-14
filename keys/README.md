# Field-DataExploration

**Setting Private Authorization Keys**:

1. Create an authorized_keys.yaml file in the  ```keys/``` folder
  ```
  touch authorized_keys.yaml
  ```

2. Add blobs keys to the file 

  Example :
  ```
  blobs:
      <container-name-1>:
          url: "https://<container-name>.table.core.windows.net/"
          sas_token: "<PRIVATE KEY 1>"
      <container-name-2>:
          url: "https://<container-name>.table.core.windows.net/"
          sas_token: "<PRIVATE KEY 2>"
  ```

3. Add Table Keys to the file

  Example :
  ```
  tables:
    <table-name-1>:
      url: "https://<container-name>.table.core.windows.net/"
      sas_token: "<PRIVATE KEY 1>"

    <table-name-2>:
      url: "https://<container-name>.table.core.windows.net/"
      sas_token: "<PRIVATE KEY 2>"

  ```