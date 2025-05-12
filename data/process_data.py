import pandas as pd

df = pd.read_csv("processed_stimuli.csv")
df.head(100).to_csv("processed_stimuli_100.csv")
# dict_keys(['fixations', 'attention_mask', 'mapped_fixations'])
# import json

# with open("fixation_records.jsonl") as f:
#     records = [json.loads(line) for line in f]

# rec = records[0]
# m = len(rec['fixations'])
# n = len(rec['fixations'][0])
# print(m, n)
# rec = records[1]

# m = len(rec['fixations'])
# n = len(rec['fixations'][0])
# print(m, n)