from mbi import Dataset, FactoredInference
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

data = Dataset.load('./data/adult_part.csv', './data/adult-domain.json')
domain = data.domain
total = data.df.shape[0]
print(data.df)
print(len(data.domain))

age = data.project(['age']).datavector()
print('age:\n', age)

epsilon = 100
sigma = np.sqrt(2.0) / epsilon

age_noize = age + np.random.laplace(loc=0, scale=sigma, size=age.size)
print('age_noize:\n', age_noize)

Iage = np.eye(age.size)

measurements = [(Iage, age_noize, sigma, 'age')]
engine = FactoredInference(domain, log=True)
model = engine.estimate(measurements, engine='MD', total=total+1)
print('model: \n', model.synthetic_data().df)

print('age:\n', model.project(['age']).datavector())
age_rounded = np.round(model.project(['age']).datavector())
print('age\n', age_rounded)

print(np.linalg.norm((np.array(age) - np.array(age_rounded)), ord=1))