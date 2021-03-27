import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

#data pre processing
df = df.drop('ID', axis=1)
