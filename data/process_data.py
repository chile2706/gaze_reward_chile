import pandas as pd

df = pd.read_csv("processed_stimuli.csv")
df.head(10).to_csv("processed_stimuli_10.csv")