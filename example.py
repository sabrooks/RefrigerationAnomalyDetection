from refrigeration_anomaly_detection.pipeline import anomaly_pipeline

import pandas as pd

import numpy as np

df = pd.read_csv("DataHvALabel.csv", low_memory=False, parse_dates=True)

omg = df.where(df.loc[:, "series"] == "Omgeving")\
    .set_index("timestamp")\
    .sort_index()

omg.to_parquet("data/omg.parquet")

X = omg.loc[:, "value"]
widths = np.arange(100, 3100, 100)

pipe = anomaly_pipeline[:-1].set_params(**{"wavelet__widths": widths}).fit(X)

print(omg.shape)
