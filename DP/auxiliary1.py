import numpy as np
import pandas as pd
import time
from tqdm._tqdm import trange
from tqdm import tqdm


def answer_queries_dict(dataset, member, queries):
    answer = {}
    for length in queries[member].keys():
        answer[length] = []
        for query in queries[member][length]:
            answer[length] += [len(query(dataset))]
    return answer


def answer_queries(dataset, member, queries):
    answer = []
    for length in queries[member].keys():
        for query in queries[member][length]:
            answer += [len(query(dataset))]
    return answer


def output_answer(answer, member, query_instance, logger=None):
    head = 0
    tail = 0
    logger.info('output_answer:%s' % answer)
    for length in query_instance.length_size[member]:
        logger.info('query_instance.length_size[member]:%s' % query_instance.length_size[member])
        tail = query_instance.length_size[member][length]
        logger.info('tail, head: %s %s' % (head, tail))
        if logger is not None:
            logger.info('Range size %d, answer %s:' % (length, str(answer[head: tail])))
        else:
            print('Range size %d, answer:' % length)
            print(str(answer[head: tail]))
        head = query_instance.length_size[member][length]


def store_answer(dynamic_tree, insertion_deletion_mechanism, ipp_instance, query_length=None, logger=None):
    logger.info('Integrated answers stores here')
    members = dynamic_tree.query_instance.queries.keys()
    indexes = range(1, len(ipp_instance.get_segment()) - 1)
    for member in members:
        logger.info('Member %s' % member)
        for index in indexes:
            logger.info('Node Index %d' % index)
            if query_length is None:
                logger.info('Dynamic Tree:')
                logger.info('Ground truth: %s' % dynamic_tree.answer_ground_truth[member][index])
                logger.info('Golden standard: %s' % dynamic_tree.answer_golden_standard[member][index])
                logger.info('Mechanism: %s' % dynamic_tree.answer_mechanism[member][index])
                logger.info('Insertion-Only Mechanism:')
                logger.info('Ground Truth: %s' % insertion_deletion_mechanism.answer_ground_truth[member][index])
                logger.info('Mechanism: %s' % insertion_deletion_mechanism.answer_mechanism[member][index])


def compare_results(answer1, answer2, measurement=1):
    # For golden standard, returns the queries' answer with pgm mechanism on single static dataset.
    # Parameter: measurement:
    # measurement=1: give the average of the difference divides the golden_standard
    # measurement=2: give the mean difference between the queries
    m_answer1 = [i + 1 for i in answer1]
    m_answer2 = [i + 1 for i in answer2]
    if measurement == 1:
        difference = np.absolute(np.array(m_answer1) - np.array(m_answer2))
        result = np.array(difference) / np.array(m_answer2)
        return np.mean(result)
    if measurement == 2:
        return (np.mean(m_answer1)) / np.mean(m_answer2)
    if measurement == 3:
        difference = np.absolute(np.array(m_answer1) - np.array(m_answer2))
        return np.sum(difference) / np.sum(m_answer1)
    return -1
