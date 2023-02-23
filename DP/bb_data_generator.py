# The file generate the dataset where contains the label 'x' and data 'y'
# For 'x', 1 for deletion, 0 for no action, and 1 for deletion
# To satisfy the sparse requirement, we let P(x=1)<<P(x=2)

import csv
import random
import pandas as pd
import json

COLUMN_NUM = 2
SIZE = 1000
UNIVERSE = 1000


def p_random(arr1, arr2):
    assert len(arr1) == len(arr2)
    assert sum(arr2) == 1
    sup_list = [len(str(i).split(".")[-1]) for i in arr2]
    top = 10 ** max(sup_list)
    new_rate = [int(i * top) for i in arr2]
    rate_arr = []
    for i in range(1, len(new_rate) + 1):
        rate_arr.append(sum(new_rate[:i]))
    rand = random.randint(1, top)
    data = None
    for i in range(len(rate_arr)):
        if rand <= rate_arr[i]:
            data = arr1[i]
            break
    return data


cond_list = []
for i in range(SIZE):
    cond_list.append({'x': p_random([0, 1], [0, 1]), 'y1': random.randint(1, UNIVERSE), 'y2': random.randint(1, UNIVERSE)})
for i in range(len(cond_list)):
    if cond_list[i]['x'] == 1:
        cond_list.append({'x': -1, 'y1': cond_list[i]['y1'], 'y2': cond_list[i]['y2']})
df = pd.DataFrame(cond_list, columns=['x', 'y1', 'y2'])
df.drop_duplicates()
print(df.info())
bb_example_s = {'x': 3, 'y1': UNIVERSE, 'y2': UNIVERSE}
with open("/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/bb_example.json", "w") as f:
    json.dump(bb_example_s, f)
df.to_csv("/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/bb_dataset.csv", index=False)
