import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as pyoff


def store_answer(answer_list, clique, query_instance, logger=None):
    for member in clique:
        logger.info('########-------- %s --------########' % member)
        for answer in answer_list:
            answer[member]
        logger.info('########-------- finish --------########')


def draw_mean_diagram(dynamic_tree, insertion_deletion_instance, query_instance, ipp_instance, figure_file_name,
                 query_length=None):
    members = query_instance.clique
    index_range = range(1, len(ipp_instance.get_segment()) - 1)
    for member in members:
        if query_instance.query_type == 'linear query':
            cur_range = query_instance.query_size
        elif query_instance.query_type == 'range query':
            length_size = 1
            if query_length is None:
                query_length = range(1, query_instance.config[member] + 1)
            for length in query_length:
                if length in list(query_instance.length_size[member].keys()):
                    length_size = length
            cur_range = query_instance.length_size[member][length_size]
        sum_1, sum_2, sum_3, sum_4, sum_5 = {}, {}, {}, {}, {}
        for index in index_range:
            sum_1[index] = np.mean(dynamic_tree.answer_ground_truth[member][index][0:cur_range])
            sum_2[index] = np.mean(dynamic_tree.answer_golden_standard[member][index][0:cur_range])
            sum_3[index] = np.mean(dynamic_tree.answer_mechanism[member][index][0:cur_range])
            sum_4[index] = np.mean(insertion_deletion_instance.answer_ground_truth[member][index][0:cur_range])
            sum_5[index] = np.mean(insertion_deletion_instance.answer_mechanism[member][index][0:cur_range])
        trace_ground_truth = go.Scatter(x=list(index_range), y=list(sum_1.values()), mode='lines+markers',
                                        name='ground truth')
        trace_golden_standard = go.Scatter(x=list(index_range), y=list(sum_2.values()), mode='lines+markers',
                                           name='golden standard')
        trace_mechanism = go.Scatter(x=list(index_range), y=list(sum_3.values()), mode='lines+markers',
                                     name='mechanism')
        trace_ground_truth_insertion = go.Scatter(x=list(index_range), y=list(sum_4.values()), mode='lines+markers',
                                                  name='ground truth for insertion mechanism')
        trace_insertion_only = go.Scatter(x=list(index_range), y=list(sum_5.values()), mode='lines+markers',
                                          name='insertion only mechanism')
        data = [trace_ground_truth, trace_golden_standard, trace_mechanism, trace_ground_truth_insertion,
                trace_insertion_only]
        figure = go.Figure(data=data)
        figure.update_layout(width=1400, height=1000, title='random query mean', xaxis_title='Index', yaxis_title='Mean')
        figure.update_xaxes(fixedrange=True)
        figure.write_image(figure_file_name + str(member) + '.jpg')


def draw_error_diagram(dynamic_tree, insertion_deletion_instance, query_instance, ipp_instance, figure_file_name,
                       query_length=None):
    members = query_instance.clique
    index_range = np.array(range(1, len(ipp_instance.get_segment()) - 1))
    answer_ground_truth = dynamic_tree.answer_ground_truth
    answer_golden_standard = dynamic_tree.answer_golden_standard
    answer_mechanism = dynamic_tree.answer_mechanism
    answer_baseline = insertion_deletion_instance.answer_mechanism
    total_width, n = 0.8, 3
    width = total_width / n
    x = index_range - (total_width - width) / 2
    for member in members:
        if query_instance.query_type == 'linear query':
            cur_range = query_instance.query_size
        elif query_instance.query_type == 'range query':
            length_size = 1
            if query_length is None:
                query_length = range(1, query_instance.config[member] + 1)
            for length in query_length:
                if length in list(query_instance.length_size[member].keys()):
                    length_size = length
            cur_range = query_instance.length_size[member][length_size]
        re1, re2, re3 = [], [], []
        ae1, ae2, ae3 = [], [], []
        aae1, aae2, aae3 = [] , [], []
        rmse1, rmse2, rmse3 = [], [], []
        for index in index_range:
            re1.append(l1_relative_error(answer_ground_truth[member][index][0: cur_range],
                                         answer_golden_standard[member][index][0: cur_range]))
            re2.append(l1_relative_error(answer_ground_truth[member][index][0: cur_range],
                                         answer_mechanism[member][index][0: cur_range]))
            re3.append(l1_relative_error(answer_ground_truth[member][index][0: cur_range],
                                         answer_baseline[member][index][0: cur_range]))
            ae1.append(l1_absolute_error(answer_ground_truth[member][index][0: cur_range],
                                         answer_golden_standard[member][index][0: cur_range]))
            ae2.append(l1_absolute_error(answer_ground_truth[member][index][0: cur_range],
                                         answer_mechanism[member][index][0: cur_range]))
            ae3.append(l1_absolute_error(answer_ground_truth[member][index][0: cur_range],
                                         answer_baseline[member][index][0: cur_range]))
            aae1.append(l1_absolute_error(answer_ground_truth[member][index][0: cur_range],
                                         answer_golden_standard[member][index][0: cur_range]) / query_instance.query_size)
            aae2.append(l1_absolute_error(answer_ground_truth[member][index][0: cur_range],
                                         answer_mechanism[member][index][0: cur_range]) / query_instance.query_size)
            aae3.append(l1_absolute_error(answer_ground_truth[member][index][0: cur_range],
                                         answer_baseline[member][index][0: cur_range]) / query_instance.query_size)
            rmse1.append(RMSE_error(answer_ground_truth[member][index][0: cur_range],
                                    answer_golden_standard[member][index][0: cur_range]))
            rmse2.append(RMSE_error(answer_ground_truth[member][index][0: cur_range],
                                    answer_mechanism[member][index][0: cur_range]))
            rmse3.append(RMSE_error(answer_ground_truth[member][index][0: cur_range],
                                    answer_baseline[member][index][0: cur_range]))
        # re_data = [go.Bar(name='golden standard', x=list(index_range), y=list(re1)),
        #            go.Bar(name='mechanism', x=list(index_range), y=list(re2)),
        #            go.Bar(name='baseline', x=list(index_range), y=list(re3))]
        # relative_error_figure = go.Figure(data=re_data)
        # relative_error_figure.update_layout(barmode='group', width=1400, height=1000)
        # relative_error_figure.update_xaxes(fixedrange=True)
        # relative_error_figure.write_image(figure_file_name + 'relative error histogram on' + str(member) + '.jpg')
        re_data = [go.Scatter(x=list(index_range), y=list(re1), mode='lines+markers', name='golden standard'),
                   go.Scatter(x=list(index_range), y=list(re2), mode='lines+markers', name='mechanism'),
                   go.Scatter(x=list(index_range), y=list(re3), mode='lines+markers', name= 'baseline')]
        relative_error_figure = go.Figure(data=re_data)
        relative_error_figure.update_layout(width=1400, height=1000, title='Relative Error Line Chart', xaxis_title='Index', yaxis_title='Error')
        relative_error_figure.update_xaxes(fixedrange=True)
        relative_error_figure.write_image(figure_file_name + 'relative error histogram on ' + str(member) + '.jpg')
        ae_data = [go.Scatter(x=list(index_range), y=list(ae1), mode='lines+markers', name='golden standard'),
                   go.Scatter(x=list(index_range), y=list(ae2), mode='lines+markers', name='mechanism'),
                   go.Scatter(x=list(index_range), y=list(ae3), mode='lines+markers', name='baseline')]
        absolute_error_figure = go.Figure(data=ae_data)
        absolute_error_figure.update_layout(width=1400, height=1000, title='Absolute Error Line Chart', xaxis_title='Index', yaxis_title='Error')
        absolute_error_figure.update_xaxes(fixedrange=True)
        absolute_error_figure.write_image(figure_file_name + 'absolute error histogram on ' + str(member) + '.jpg')
        aae_data = [go.Scatter(x=list(index_range), y=list(aae1), mode='lines+markers', name='golden standard'),
                    go.Scatter(x=list(index_range), y=list(aae2), mode='lines+markers', name='mechanism'),
                    go.Scatter(x=list(index_range), y=list(aae3), mode='lines+markers', name='baseline')]
        average_absolute_error_figure = go.Figure(data=aae_data)
        average_absolute_error_figure.update_layout(width=1400, height=1000, title='Average Absolute Error Line Chart', xaxis_title='Index', yaxis_title='Error')
        average_absolute_error_figure.update_xaxes(fixedrange=True)
        average_absolute_error_figure.write_image(figure_file_name + 'average absolute error histogram on ' + str(member) + '.jpg')
        rmse_data = [go.Scatter(x=list(index_range), y=list(rmse1), mode='lines+markers', name='golden standard'),
                     go.Scatter(x=list(index_range), y=list(rmse2), mode='lines+markers', name='mechanism'),
                     go.Scatter(x=list(index_range), y=list(rmse3), mode='lines+markers', name='baseline')]
        rmse_figure = go.Figure(data=rmse_data)
        rmse_figure.update_layout(width=1400, height=1000, title='RMSE Line Chart', xaxis_title='Index', yaxis_title='Error')
        rmse_figure.update_xaxes(fixedrange=True)
        rmse_figure.write_image(figure_file_name + 'RMSE diagram on ' + str(member) + '.jpg')


def l1_relative_error(ground_truth, mechanism):
    if np.sum(ground_truth) == 0:
        return 1
    else:
        return np.linalg.norm(np.array(ground_truth) - np.array(mechanism), 1) / np.linalg.norm(ground_truth, 1)


def l1_absolute_error(ground_truth, mechanism):
    return np.linalg.norm(np.array(ground_truth) - np.array(mechanism), 1)


def RMSE_error(ground_truth, mechanism):
    mse = np.square(np.linalg.norm(np.array(ground_truth) - np.array(mechanism), 2)) / len(ground_truth)
    return np.sqrt(mse)
