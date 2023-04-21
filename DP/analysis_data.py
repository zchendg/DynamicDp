import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        plt.title('Testing on %s' % member)
        plt.plot(index_range, sum_1.values())
        plt.plot(index_range, sum_2.values())
        plt.plot(index_range, sum_3.values())
        plt.plot(index_range, sum_4.values())
        plt.plot(index_range, sum_5.values())
        plt.xticks(index_range)
        plt.legend(['ground truth', 'golden standard', 'mechanism', 'ground truth for ground' ,'insertion only mechanism'], loc='upper left')
        plt.savefig(figure_file_name + str(member) + '.jpg')
        plt.cla()
