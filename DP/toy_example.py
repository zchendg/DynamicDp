from mbi import Dataset, Domain, FactoredInference
import numpy as np

domain = Domain(['A','B','C'], [2,3,4])
data = Dataset.synthetic(domain, 10)
print(data.df)
print(data.project(['A', 'B']).datavector())

epsilon = 1
sigma = 1.0 / epsilon
ab = data.project(['A','B']).datavector()
bc = data.project(['B','C']).datavector()
print('ab\n', ab)
yab = ab + np.random.laplace(loc=0, scale=(sigma/np.sqrt(2)), size=ab.size)
# ybc = bc + np.random.laplace(loc=0, scale=sigma, size=bc.size)
print('yab\n', yab)

Iab = np.eye(ab.size)
Ibc = np.eye(bc.size)
# measurements = [(Iab, yab, sigma, ('A','B')), (Ibc, ybc, sigma, ('B','C'))]
# measurements = [(Iab, yab, sigma, ('A','B'))]
measurements = [(Iab, yab, sigma, ('A','B'))]
engine = FactoredInference(domain, log=True)
model = engine.estimate(measurements, engine='MD')
print('model\n', model.synthetic_data().df)

ab2 = model.project(['A','B']).datavector()
bc2 = model.project(['B','C']).datavector()
ac2 = model.project(['A','C']).datavector()
print('ab2\n', ab2)
print('bc2\n', bc2)
print('ac2\n', ac2)

sum = 0
for t in ab2:
    sum += t
print(t)