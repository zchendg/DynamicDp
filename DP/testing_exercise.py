import numpy as np
import pandas as pd
import json
from approximation_instance import ApproximationInstance
from mbi import Domain, Dataset
from query import Query
from catboost.datasets import adult
from my_logger import Logger

domain = "/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/data/adult-domain.json"
config = json.load(open(domain))
domain = Domain(config.keys(), config.values())
df = pd.read_csv('data/adult.csv', sep=',')
df = df.iloc[0:100]
print(config)
print(df[df.eval('%d<%s<%d' % (0, '`income>50K`', 100))])

query = Query(config, 5, 10)
print(query.config)
answer = 0
for member in query.queries:
    print(member)
    for query_instance in query.queries[member]:
        answer += len(query_instance(df))
print(answer)


a1 = np.random.normal(size=(1000000,100))
b1 = np.random.normal(size=(1000000,100))
mses = ((a1-b1)**2).mean(axis=1)
print(mses)
a = np.array([1,2,3])
b = np.array([4,5,6])
mse = ((a-b)**2).mean()
print(mse)