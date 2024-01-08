import pandas as pd
import numpy as np

from peakdetect import peakdet

example_file = \
    '../GridWalking-wyc-2023-10-30/new_analyzed/AD/3-4M/AD688_2023-09-11_11-06-19DLC_resnet50_GridWalkingDec12shuffle1_50000_filtered.h5'

df = pd.read_hdf(example_file)

body_parts = pd.unique(df.columns.get_level_values(1))