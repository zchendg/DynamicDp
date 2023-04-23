import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyoff


def store_answer(answer_list, clique, query_instance, logger=None):
    for member in clique:
        logger.info('########-------- %s --------########' % member)
        for answer in answer_list:
            answer[member]
        logger.info('########-------- finish --------########')


def draw_diagram(dynamic_tree, insertion_deletion_instance, query_instance, ipp_instance, figure_file_name,
                 query_length=1):
    members = query_instance.clique
    index_range = range(1, len(ipp_instance.get_segment()) - 1)
    for member in members:
        cur_range = query_instance.length_size[member][query_length]
        sum_1, sum_2, sum_3, sum_4, sum_5 = {}, {}, {}, {}, {}
        for index in index_range:
            sum_1_temp, sum_2_temp, sum_3_temp, sum_4_temp, sum_5_temp = 0, 0, 0, 0, 0
            for i in range(cur_range):
                sum_1_temp += dynamic_tree.answer_ground_truth[member][index][i] * i
                sum_2_temp += dynamic_tree.answer_golden_standard[member][index][i] * i
                sum_3_temp += dynamic_tree.answer_mechanism[member][index][i] * i
                sum_4_temp += insertion_deletion_instance.answer_ground_truth[member][index][i] * i
                sum_5_temp += insertion_deletion_instance.answer_mechanism[member][index][i] * i
            sum_1[index] = sum_1_temp / sum(dynamic_tree.answer_ground_truth[member][index][0:cur_range])
            sum_2[index] = sum_2_temp / sum(dynamic_tree.answer_golden_standard[member][index][0:cur_range])
            sum_3[index] = sum_3_temp / sum(dynamic_tree.answer_mechanism[member][index][0:cur_range])
            sum_4[index] = sum_4_temp / sum(insertion_deletion_instance.answer_ground_truth[member][index][0:cur_range])
            sum_5[index] = sum_5_temp / sum(insertion_deletion_instance.answer_mechanism[member][index][0:cur_range])
        trace_ground_truth = go.Scatter(x=list(index_range), y=list(sum_1.values()), mode='lines+markers', name='ground truth')
        trace_golden_standard = go.Scatter(x=list(index_range), y=list(sum_2.values()), mode='lines+markers', name='golden standard')
        trace_mechanism = go.Scatter(x=list(index_range), y=list(sum_3.values()), mode='lines+markers', name='mechanism')
        trace_ground_truth_insertion = go.Scatter(x=list(index_range), y=list(sum_4.values()), mode='lines+markers', name='ground truth for insertion mechanism')
        trace_insertion_only = go.Scatter(x=list(index_range), y=list(sum_5.values()), mode='lines+markers', name='insertion only mechanism')
        data = [trace_ground_truth, trace_golden_standard, trace_mechanism, trace_ground_truth_insertion, trace_insertion_only]
        figure = go.Figure(data=data)
        figure.write_image(figure_file_name + str(member) + '.jpg')
        # plt.title('Testing on %s' % member)
        # plt.plot(index_range, sum_1.values())
        # plt.plot(index_range, sum_2.values())
        # plt.plot(index_range, sum_3.values())
        # plt.plot(index_range, sum_4.values())
        # plt.plot(index_range, sum_5.values())
        # plt.xticks(index_range)
        # plt.legend(
        #     ['ground truth', 'golden standard', 'mechanism', 'ground truth for ground', 'insertion only mechanism'],
        #     loc='upper left')
        # plt.savefig(figure_file_name + str(member) + '.jpg')
        # plt.cla()


# def draw_diagram_error(dynamic_tree, insertion_deletion_instance, query_instance, ipp_instance, figure_file_name,
#                        query_length=1):
#     members = query_instance.clique
#     index_range = np.array(range(1, len(ipp_instance.get_segment()) - 1))
#     answer_ground_truth = dynamic_tree.answer_ground_truth
#     answer_golden_standard = dynamic_tree.answer_golden_standard
#     answer_mechanism = dynamic_tree.answer_mechanism
#     answer_baseline = insertion_deletion_instance.answer_mechanism
#     total_width, n = 0.8, 3
#     width = total_width / n
#     x = index_range - (total_width - width) / 2
#     for member in members:
#         cur_range = query_instance.length_size[member][query_length]
#         a1, a2, a3 = [], [], []
#         for index in index_range:
#             a1.append(l1_error(answer_ground_truth[member][index][0: cur_range],
#                                answer_golden_standard[member][index][0: cur_range]))
#             a2.append(l1_error(answer_ground_truth[member][index][0: cur_range],
#                                answer_mechanism[member][index][0: cur_range]))
#             a3.append(l1_error(answer_ground_truth[member][index][0: cur_range],
#                                answer_baseline[member][index][0: cur_range]))
#         plt.title('Histogram of %s' % member)
#         plt.bar(x, a1, width=width, label='golden standard')
#         plt.bar(x + width, a2, width=width, label='mechanism')
#         plt.bar(x + 2 * width, a3, width=width, label='baseline')
#         plt.legend()
#         plt.savefig(figure_file_name + 'error histogram on' + str(member) + 'jpg')
#         plt.cla()


def l1_error(ground_truth, mechanism):
    if np.sum(ground_truth) == 0:
        return 1
    else:
        return np.linalg.norm(np.array(ground_truth) - np.array(mechanism), 1) / np.sum(np.linalg.norm(ground_truth))
