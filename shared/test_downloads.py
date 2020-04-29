#!/usr/bin/env python
"""
Direct Landsat Downloads

2020-04-29 12:15:51
"""
import pandas as pd
import numpy as np
import pathlib
import urllib.request
import ee
import subprocess



output_dir = pathlib.Path("../data/")
index_path = pathlib.Path(output_dir, "index.csv.gz")


if not index_path.exists():
    webpath = "https://storage.googleapis.com/gcp-public-data-landsat/index.csv.gz"
    urllib.request.urlretrieve(webpath, index_path)

index = pd.read_csv(index_path, compression="gzip")
index07 = index[index.SPACECRAFT_ID == "LANDSAT_7"]
subprocess.check_output(["gsutil", "-m", "cp", "-r", index07.BASE_URL.values[-1], output_dir])

# can we search for a particular scene?
# ids = "LE07_134041_20051103"
