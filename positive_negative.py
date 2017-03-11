import os
from os import listdir
from os.path import join, realpath

import numpy as np
import pandas as pd


path = realpath('')
stages = join(path,'stage1_labels.csv')
print(stages)


stages_raw_df = pd.read_csv(stages, low_memory=False);
stages_raw_df

for file in listdir(join(path,'sample_images')):
    if ("._" in file):
        continue
    print(listdir(join(path,'sample_images', file)))
    