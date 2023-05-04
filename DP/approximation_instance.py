from mbi import Dataset, FactoredInference
import numpy as np
import pandas as pd
from scipy import sparse


class ApproximationInstance:
    def __init__(self, df, domain, epsilon, cliques, source='PATH', iters=2500):
        if source == 'PATH':
            self.data = Dataset.load(df, domain)
        else:
            self.data = Dataset(df, domain)
        self.domain = self.data.domain
        self.total = self.data.df.shape[0]
        self.approximated_data, self.model = self.estimate_data_distribution(epsilon, cliques, iters)

    def estimate_data_distribution(self, epsilon, cliques, iters=2500):
        # Need to know what is the differential budget for this algorithm
        epsilon
        sigma = np.sqrt(2) / epsilon
        measurements = []
        for cl in cliques:
            x = self.data.project(cl).datavector()
            y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
            I = sparse.eye(x.size)
            measurements.append((I, y, sigma, cl))
        engine = FactoredInference(self.domain, log=True, iters=iters)
        # model = engine.estimate(measurements, total=self.total)
        model = engine.estimate(measurements, total=self.total + 1)
        return model.synthetic_data(), model
