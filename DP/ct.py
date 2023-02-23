import pandas as pd
import numpy as np
import datetime
from basic_counting import BasicCounting
from ipp import IPP


print("-- Starting @ %ss" % datetime.datetime.now())
df = pd.read_csv('/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/simple DP dataset.csv', sep=',')
# print(df.head(20))

# BC = BasicCounting(df)
# print(BC.update_counting())

# arr = [np.random.randint(low=0, high=n, size=20) for n in [2,3,4,5]]
# values = np.array(arr).T
# df = pd.DataFrame(values, columns = ['a', 'b', 'c', 'd'])
# print(df)

# bins = [range(n+1) for n in [2, 3, 4, 5]]
# print(bins)

ipp_instance = IPP(df, 5, 0.05)
print(ipp_instance)
for t in range(len(df)):
    ipp_instance.update_segment(t)
print(ipp_instance.get_segment())