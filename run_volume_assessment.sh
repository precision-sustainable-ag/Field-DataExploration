#!/bin/bash

python FIELD_REPORT.py general.task=wir_table_generator &
python FIELD_REPORT.py general.task=wir_blob_data_generator &
