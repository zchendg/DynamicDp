import numpy as np
import pandas as pd
from node import Node
from query import Query
from approximation_instance import ApproximationInstance
from mbi import Domain, Dataset
from basic_counting import BasicCounting


class DynamicTree:
    def __init__(self, config):
        self.node_list = [Node(0, config.keys())]
        self.config = config
        self.domain = Domain(config.keys(), config.values())

    def insert_item(self, index, item):
        self.node_list[index - 1].add_item(item)

    def create_node(self):
        # Create a node at the end of the tree in the sequential time order
        self.node_list.append(Node(len(self.node_list), self.config.keys()))

    def df_after_closest_left_ancestor(self, index, current_df):
        set_diff_df = current_df.data
        for node in self.query_nodes(index):
            set_diff_df = pd.concat([set_diff_df, node.df, node.df]).drop_duplicates(keep=False)
        return set_diff_df

    def create_leftmost_node(self, cur_data, delete_data):
        self.create_node()
        self.node_list[-1].add_items(cur_data.data)
        self.node_list[-1].add_items(delete_data.data, delete_df=True)

    def create_internal_node(self, cur_data, delete_data):
        self.create_node()
        diff_df = self.df_after_closest_left_ancestor(len(self.node_list) - 1, cur_data)
        self.node_list[-1].add_items(diff_df)
        self.node_list[-1].add_items(delete_data.data, delete_df=True)

    def find_far_left_ancestor_index(self, index, height):
        if index / (2 ** height) % 4 == 1:
            return self.find_far_left_ancestor_index(index + 2 ** height, height + 1)
        elif index / (2 ** height) % 4 == 3:
            return index - 2 ** height

    def is_two_power(self, n):
        if n == 1:
            return True
        res = 1
        while res < n:
            res = res << 1
        if res == n:
            return True
        else:
            return False

    # Below is the function serves to queries
    def query_nodes(self, index):
        nodes = [self.node_list[index]]
        if self.is_two_power(index):
            return nodes
        else:
            return nodes + self.query_nodes(
                self.find_far_left_ancestor_index(self.node_list[index].index, self.node_list[index].height))

    # Implement testing function in the data structure
    def testing(self, ipp_instance, column_number=1, each_query_size=10, epsilon=1, delta=0, beta=0.05, iteration=500,
                logger=None):
        for index in range(1, len(ipp_instance.get_segment()) - 1):
            logger.info('-------- Testing on node %d Started --------' % index)
            self.testing_index(index, column_number, each_query_size, epsilon, delta, beta, iteration, logger)
            logger.info('-------- Testing on node %d Finished --------' % index)

    # Implement testing for at particular position, this function
    def testing_index(self, index, column_number=1, each_query_size=100, epsilon=1, delta=0, beta=0.05, iteration=500,
                      logger=None):
        query_instance = Query(self.config, column_number, each_query_size)
        query_nodes = self.query_nodes(index)
        query_nodes.reverse()
        logger.info(
            'At node with index %d, we implement queries on cliques %s:' % (index, query_instance.queries.keys()))
        for member in query_instance.queries.keys():
            # answer = self.answer_queries(query_nodes, index, query_instance.queries[member], member, epsilon,
            #                                   delta,
            #                                   iteration)
            answer_ground_truth = self.answer_queries_ground_truth(query_nodes, index, query_instance.queries[member],
                                                                   logger=logger)
            answer_golden_standard = self.answer_queries_golden_standard(query_nodes, index,
                                                                         query_instance.queries[member], member,
                                                                         epsilon, delta, iteration, logger=logger)
            answer_mechanism = self.answer_queries_mechanism(query_nodes, index, query_instance.queries[member], member,
                                                             epsilon, delta, beta, iteration, logger)
            logger.info('The testing is implemented at %s' % member)
            logger.info('Ground truth: gives answer: \n%s' % np.array(answer_ground_truth))
            logger.info('Golden standard: gives answer: \n%s' % np.array(answer_golden_standard))
            logger.info('Mechanism: gives answer: \n%s' % np.array(answer_mechanism))
            logger.info("Mean Square Error of ground truth and golden standard: %s" % self.mse(answer_ground_truth,
                                                                                               answer_golden_standard))
            logger.info(
                "Mean Square Error of ground truth and mechanism: %s" % self.mse(answer_ground_truth, answer_mechanism))
            logger.info("Mean Square Error of golden standard and mechanism: %s" % self.mse(answer_golden_standard,
                                                                                            answer_mechanism))
            logger.info('Measurement1: ' + str(
                self.compare_results(answer_ground_truth, answer_golden_standard, measurement=1)))
            logger.info('Measurement2: ' + str(
                self.compare_results(answer_ground_truth, answer_golden_standard, measurement=2)))

    def answer_queries(self, nodes, cur_index, queries, member, epsilon=1, delta=0, iteration=500):
        epsilon_budge = {node: 6 * epsilon / (np.square(np.pi * (node.height + 1))) for node in nodes}
        delta_budge = {node: 6 * delta / (np.square(np.pi * (node.height + 1))) for node in nodes}
        answer = [0] * len(queries)
        for node in nodes:
            queries_answer = self.answer_node_queries_mechanism(node, cur_index, queries, member,
                                                                epsilon_budge[node],
                                                                delta_budge[node], iteration)
            answer = np.array(answer) + np.array(queries_answer)
        return answer

    # For ground truth, returns the queries' answer for original data.
    def answer_queries_ground_truth(self, nodes, cur_index, queries, logger=None):
        Dv_list = []
        for node in nodes:
            Dv = pd.DataFrame(columns=self.config.keys())
            delete_df = pd.DataFrame(columns=self.config.keys())
            for index in range(cur_index + 1):
                if index == node.index:
                    Dv = node.df
                elif node.index < index <= cur_index:
                    delete_df = pd.merge(
                        pd.concat([delete_df, self.node_list[index].delete_df]).drop_duplicates(keep=False), Dv,
                        how='inner')
                    Dv = pd.concat([Dv, delete_df, delete_df]).drop_duplicates(keep=False)
            Dv_list += [Dv]
        Dataset_r = pd.concat(Dv_list).drop_duplicates(keep=False)
        # logger.info('Ground truth: the original dataset is: \n%s ' % Dataset_r)
        answer_ground_truth = []
        for query in queries:
            answer_ground_truth += [len(query(Dataset_r))]
        logger.info('Ground truth: gives answer: \n%s' % np.array(answer_ground_truth))
        return np.array(answer_ground_truth)

    # For golden standard, returns the queries' answer for golden standard (Approximated data)
    def answer_queries_golden_standard(self, nodes, cur_index, queries, member, epsilon=1, delta=0, iteration=500,
                                       logger=None):
        Dv_list = []
        for node in nodes:
            Dv = pd.DataFrame(columns=self.config.keys())
            delete_df = pd.DataFrame(columns=self.config.keys())
            for index in range(cur_index + 1):
                if index == node.index:
                    Dv = node.df
                elif node.index < index <= cur_index:
                    delete_df = pd.merge(
                        pd.concat([delete_df, self.node_list[index].delete_df]).drop_duplicates(keep=False), Dv,
                        how='inner')
                    Dv = pd.concat([Dv, delete_df, delete_df]).drop_duplicates(keep=False)
            Dv_list += [Dv]
        Dataset_r = pd.concat(Dv_list).drop_duplicates(keep=False)
        # Modification
        approximate_instance = ApproximationInstance(Dataset_r, self.domain, epsilon, [member], 'Data', iteration)
        # logger.info('Golden standard: the original dataset is: \n%s ' % Dataset_r)
        data = Dataset(Dataset_r, self.domain)
        # logger.info('Golden standard: the approximated dataset is: \n%s ' % approximate_instance.approximated_data.df)
        logger.info('Golden standard: data vector for original dataset is: \n%s' % data.project(['age']).datavector())
        logger.info('Golden standard: data vector for approximate dataset is: \n%s' % np.round(
            approximate_instance.model.project(['age']).datavector()))
        logger.info('epsilon %s' % epsilon)
        # Modification
        answer_golden_standard = []
        for query in queries:
            # Modification
            answer_golden_standard += [len(query(approximate_instance.approximated_data.df))]
            # Modification
        logger.info('Golden standard: gives answer: \n%s' % np.array(answer_golden_standard))
        return np.array(answer_golden_standard)

    # For new algorithm, returns the queries' answer for the new algorithm
    def answer_queries_mechanism(self, nodes, cur_index, queries, member, epsilon=1, delta=0, beta=0.05, iteration=500,
                                 logger=None):
        epsilon_budget = {node: 6 * epsilon / (np.square(np.pi * (node.height + 1))) for node in nodes}
        delta_budget = {node: 6 * delta / (np.square(np.pi * (node.height + 1))) for node in nodes}
        answer_mechanism = [0] * len(queries)
        for node in nodes:
            answer_mechanism_node = self.answer_node_queries_mechanism(node, cur_index, queries, member, epsilon_budget[node], delta_budget[node],
                                                                       beta, iteration, logger)
            answer_mechanism = np.array(answer_mechanism) + np.array(answer_mechanism_node)
        logger.info('Mechanism: gives answer \n%s' % answer_mechanism)
        return np.array(answer_mechanism)

    def answer_node_queries_mechanism(self, node, cur_index, queries, member, epsilon=1, delta=0, beta=0.05,
                                      iteration=500, logger=None):
        logger.info('-------- New Mechanism answer node query on node %s starts --------' % node.index)
        # Initiate delete_df to store the data in the deletion-only problem
        tilde_n_v = 0
        Dv = pd.DataFrame(columns=self.config.keys())
        delete_df = pd.DataFrame(columns=self.config.keys())
        # These variables store the query answer for the approximated dataset
        Dv_answer = []
        D_sj_answer = []
        answer_mechanism = []
        for index in range(cur_index + 1):
            if index == node.index:
                Dv = node.df
                r = 1
                logger.info('Mechanism: \n%s' % Dv)
                # Epsilon_r might be modified, as the number of restarts is fixed
                if len(Dv) == 0:
                    Dv_answer = [0] * len(queries)
                    answer_mechanism = Dv_answer
                    continue
                elif len(Dv) == 1:
                    for query in queries:
                        Dv_answer = [len(query(Dv))]
                        answer_mechanism = Dv_answer
                        continue
                epsilon_r = max(3 * epsilon / (2 * np.square(np.pi * r)), epsilon / np.log2(len(Dv)))
                delta_r = max(2 * delta / (np.square(np.pi * r)), delta / np.log2(len(Dv)))
                tilde_n_v = len(Dv) + np.random.laplace(loc=0, scale=1 / epsilon_r)
                logger.info('Mechanism: epsilon during the algorithm %f' % epsilon_r)
                approximation_instance = ApproximationInstance(Dv, self.domain, epsilon_r, [member], 'Data', iteration)
                for query in queries:
                    Dv_answer += [len(query(approximation_instance.approximated_data.df))]
                    answer_mechanism = Dv_answer
            elif node.index < index <= cur_index:
                delete_df = pd.merge(
                    pd.concat([delete_df, self.node_list[index].delete_df]).drop_duplicates(keep=False), Dv,
                    how='inner')
                tilde_n_del = len(delete_df)
                # The error bound of MBC at time sj
                alpha_BC_sj = (1 / epsilon) * (np.log2(self.node_list[index].sj) ** 1.5) * np.log2(1 / beta)
                if tilde_n_del > (tilde_n_v / 2 + 2 * alpha_BC_sj):
                    # Remove all augmented items from D(v)
                    Dv = pd.concat([Dv, delete_df, delete_df]).drop_duplicates(keep=False)
                    r = r + 1
                    epsilon_r = max(3 * epsilon / (2 * np.square(np.pi * r)), epsilon / np.log2(len(Dv)))
                    delta_r = max(2 * delta / (np.square(np.pi * r)), delta / np.log2(len(Dv)))
                    tilde_n_v = len(Dv) + np.random.laplace(loc=0, scale=1 / epsilon_r)
                    # Run(epsilon_r, delta_r)-DP M(D(v)) to release Q(D(v))
                    approximation_instance = ApproximationInstance(Dv, self.domain, epsilon_r, [member], 'Data',
                                                                   iteration)
                    # Re-publish the query answer for Dv
                    Dv_answer = []
                    for query in queries:
                        Dv_answer += [len(query(approximation_instance.approximated_data.df))]
                    # When this condition happens, we all the Q(D_t(v)) will return 0, hence just
                    if tilde_n_v < (2 * alpha_BC_sj):
                        answer_mechanism = np.array(Dv_answer)
                        return np.array(answer_mechanism)
                elif index == cur_index:
                    approximation_instance_delete = ApproximationInstance(delete_df, self.domain, epsilon_r, [member],
                                                                          'Data', iteration)
                    for query in queries:
                        D_sj_answer += [len(query(approximation_instance_delete.approximated_data.df))]
                    answer_mechanism = np.array(Dv_answer) - np.array(D_sj_answer)
                    return np.array(answer_mechanism)
        logger.info('-------- New Mechanism answer node query on node %s finished --------' % node.index)
        return np.array(answer_mechanism)

    def compare_results(self, answer1, answer2, measurement=1):
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
        return -1

    def return_golden_standard(self, queries, dataset, approximate_instance=None, logger=None):
        answer_golden_standard = []
        for query in queries:
            answer_golden_standard += [len(query(approximate_instance.approximated_data.df))]
        logger.info('Golden standard: static mechanism data gives answer: \n%s' % answer_golden_standard)
        return np.array(answer_golden_standard)

    def mse(self, array1, array2):
        return ((np.array(array1) - np.array(array2)) ** 2).mean()
