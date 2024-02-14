#!/bin/bash

parent_directory="$(pwd)/volume_assessment"

cd "$parent_directory" || exit

# Run the Blob Container volume assessment
python3 wir_blob_data_generator.py &

# Run the Blob Container volume assessment
python3 wir_table_generator.py &
