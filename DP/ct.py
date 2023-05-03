import pandas as pd
import json
import numpy as np
from approximation_instance import ApproximationInstance
from mbi import Domain, Dataset
from query import Query
import plotly.graph_objs as go
import auxiliary1
import analysis_data_1
from my_logger import Logger

data = pd.read_csv('./data/adult.csv')
config = json.load(open('./data/adult-domain.json'))
domain = Domain(config.keys(), config.values())
epsilon = 1
clique = ['age']
member = 'age'
iteration = 1000
query_instance = Query(config, query_type='linear query', random_query=True, query_size=100)
error = []
index_range = list(range(1, 40000, 10000))

for size in [1]:
    data_sized = data[0:size]
    approximation_instance = ApproximationInstance(data_sized, domain, epsilon, clique, 'Data', iteration)
    answer_ground_truth = auxiliary1.answer_queries(query_instance.query_type, data_sized, member, query_instance.queries)
    answer_pgm = auxiliary1.answer_queries(query_instance.query_type, approximation_instance.approximated_data.df, member, query_instance.queries)
    error += [analysis_data_1.l1_absolute_error(answer_ground_truth, answer_pgm)]
figure_data = [go.Scatter(x=index_range, y=error, mode='lines+markers', name='error')]
figure = go.Figure(data=figure_data)
figure.update_xaxes(fixedrange=True)
figure.write_image('error diagram.jpg')
print(error)
