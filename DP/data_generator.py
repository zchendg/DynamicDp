# The file generate the dataset where contains the label 'x' and data 'y'
# For 'x', 1 for deletion, 0 for no action, and 1 for deletion
# To satisfy the sparse requirement, we let P(x=1)<<P(x=2)

import csv
import random
import pandas as pd

SIZE = 1000
UNIVERSE = 10000


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


# with open('simple DP.csv', 'w+', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['x', 'y'])
#     insert_list = []
#     for i in range(SIZE):
#         insert_list.append([p_random([0, 1], [0.99, 0.01]), random.randint(1, UNIVERSE)])
#         writer.writerow(insert_list[i])
#     reader = csv.DictReader(file, fieldnames=['x', 'data'])
#     # Shuffle the list
#     random.shuffle(insert_list)
#     print(type(insert_list[0][0]))
#     for i in range(len(insert_list)):
#         if(insert_list[i][0] == 1):
#             writer.writerow([-1, insert_list[i][1]])

cond_list = []
for i in range(SIZE):
    cond_list.append({'x': p_random([0, 1], [0, 1]), 'y': random.randint(1, UNIVERSE)})
for i in range(len(cond_list)):
    if cond_list[i]['x'] == 1:
        cond_list.append({'x': -1, 'y': cond_list[i]['y']})
df = pd.DataFrame(cond_list, columns=['x', 'y'])
print(df.info())
df.to_csv("/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/simple DP dataset.csv", index=False)
