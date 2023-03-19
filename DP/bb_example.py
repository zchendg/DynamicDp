from mbi import Dataset, FactoredInference
import numpy as np
import pandas as pd
from scipy import sparse

data = Dataset.load('/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/data/bb_dataset.csv',
                    '/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/data/bb_example.json')
domain = data.domain
total = data.df.shape[0]

print(domain)

y1 = data.project(['y1']).datavector()
print('y1:\n', y1)

epsilon = np.sqrt(2)
sigma = np.sqrt(2.0) / epsilon

yy1 = y1 + np.random.laplace(loc=0, scale=sigma, size=y1.size)
print('yy1:\n', yy1)

Iy1 = np.eye(y1.size)

measurements = [(Iy1, yy1, sigma, ['y1'])]
engine = FactoredInference(domain)
model = engine.estimate(measurements, engine='MD')
print('model:\n', model.synthetic_data().df)
