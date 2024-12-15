import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# TEMP FILE TO CHECK THE DATA. NOT PART OF THE FINAL PROJECT

df = pd.read_csv("/home/mees/Desktop/forestfires_usa/dataset/texas_fires_drought.csv")

print(df['STAT_CAUSE_DESCR'].value_counts())

df = df.sort_values(by='FIRE_SIZE', ascending=False)

print(df['FIRE_SIZE'].isna().sum())

print(df['FIRE_SIZE_CLASS'].isna().sum())
print(df['STAT_CAUSE_DESCR'].isna().sum())


print(df['FIPS'].value_counts())

print(df[df['FIPS'] == 48147]['STAT_CAUSE_DESCR'].value_counts())


