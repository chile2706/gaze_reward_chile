import pandas as pd

df = pd.read_csv("processed_stimuli.csv")
df.head(100).to_csv("processed_stimuli_100.csv")
# dict_keys(['fixations', 'attention_mask', 'mapped_fixations'])
# import json

# with open("fixation_records.jsonl") as f:
#     records = [json.loads(line) for line in f]

# n = len(records)
# for i in range(10):
#     rec = records[i]
#     m = len(rec['fixations'])
#     n = len(rec['fixations'][0])
    
#     m1 = len(rec['attention_mask'])
#     n1 = len(rec['attention_mask'][0])
#     print(m, n, m1, n1)
#     print(rec["sentences"])

# # 1 194 1 194
# # ['I want to make crust for pizza dough.   Here’s how to do it.  To make the crust you’ll need a large bowl and a pastry cloth, and you’ll need to start by mixing together the flour and salt.  Then you can gradually add in the butter, while mixing.  When the dough is ready, the butter should be completely incorporated, and you’ll have a dough that feels soft but stiff, and slightly warm.   I wonder if I should knead the dough using my electric bread maker.  My hands are not strong enough.   Why don’t you try kneading it by hand first?  If you’re just looking for something fast and easy, then kneading the dough isn’t necessary.  You’re also free to knead it as much as you want, if you want to.']

# # df = pd.read_csv("processed_stimuli.csv")
# # records = df.to_dict(orient="records")
# # print(records[449])
