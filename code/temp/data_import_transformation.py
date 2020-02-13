# Imports
import os

import numpy as np
import pandas as pd
from azure.storage.blob import BlockBlobService
from azure.storage.blob import ContentSettings

from utils import *

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("precision", 4)

# Azure config
block_blob_service = BlockBlobService(account_name="bigdatamigration", account_key="c6BqCdoYT78gZ/iOonKfQKlxxFucEZ8Vvwp1UWGtQEONmX4pAG27Fqzt6io62eI2pRH4UVvPJrd6u9kFJLrZSQ==")
block_blob_service.create_container("sinkstage")

# Load data
df = pd.read_csv(filepath_or_buffer="../../data/raw/winequality-red.csv", sep=";")
df = df.drop("quality", axis=1)
# Loop
for i in range(5000):
    x = df.sample(1)
    location = np.random.randint(0,9)
    sample_time = random_date("20190101_000000", "20191231_235959", np.random.random())
    # sample_time = time.strftime("%Y%m%d_%H%M%S") # for current time
    path = os.path.join("../../data//processed/", sample_time + "_" +
                        str(location) + ".json")
    # Randomly change instance values
    for j in range(11):
        x.iloc[0,j] = x.iloc[0,j] + np.random.normal(0, 0.05) * x.iloc[0,j]
    # Save to *.json
    x.to_json(
        path_or_buf=path,
        orient="records",
    )
    # Push to Blob Storage
    block_blob_service.create_blob_from_path(
        container_name="sinkstage",
        blob_name=os.path.basename(path),
        file_path=path,
        content_settings=ContentSettings(content_type="application/JSON")
    )
