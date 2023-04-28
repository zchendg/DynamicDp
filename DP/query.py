import random
import numpy as np


class Query:
    def __init__(self, config, query_type='range query', random_query=False, column_number=2, query_size=10, logger=None):
        # self.clique = random.sample(config.keys(), column_number)
        self.clique = ['age']
        self.config = config
        assert query_type in ['range query', 'linear query']
        self.query_type = query_type
        self.column_number = column_number
        self.query_size = query_size
        if query_type == 'linear query':
            self.parameters = self.generate_linear_query_parameters(query_size)
            self.queries = self.create_linear_queries()
        elif query_type == 'range query' and random_query:
            self.parameters = self.generate_random_range_query_parameters(query_size)
            self.length_size = self.random_get_length_size()
            self.queries = self.create_random_range_queries()
        elif query_type == 'range query' and (not random_query):
            self.parameters = self.generate_non_random_range_query_parameters()
            self.length_size = self.non_random_get_length_size()
            self.queries = self.create_non_random_range_queries()
        self.store_query_info(logger)

    # Serves to the range query that generating queries randomly
    #
    #
    #
    def generate_random_range_query_parameters(self, query_size, threshold=1):
        parameters = {}
        for member in self.clique:
            parameters[member] = {}
            query_parameter_list = []
            for count in range(query_size):
                # query_bounds = sorted(random.sample(range(self.config[member]), 2))
                query_bounds = self.generate_random_range_query_bound(member, int((1/3)*self.config[member]))
                query_parameter_list += [query_bounds]
            query_parameter_dict = self.random_parameter_rearrange(query_parameter_list)
            parameters[member] = query_parameter_dict
        return parameters

    def generate_random_range_query_bound(self, member, threshold=1):
        query_bounds = sorted(random.sample(range(self.config[member]), 2))
        counter = 0
        while counter < 50:
            if (query_bounds[1] - query_bounds[0] + 1) >= threshold:
                return query_bounds
            else:
                query_bounds = sorted(random.sample(range(self.config[member]), 2))
                counter + 1
        return query_bounds

    # This function returns the dict that contains the length to the query parameters
    def random_parameter_rearrange(self, query_list):
        length_list = []
        for pair in query_list:
            number = (pair[1] - pair[0] + 1)
            if number not in length_list:
                length_list.append(number)
        length_list.sort()
        query_dict = {}
        for length in length_list:
            query_dict[length] = []
            for pair in query_list:
                if (pair[1] - pair[0] + 1) == length:
                    query_dict[length] += [pair]
        return query_dict

    # Store size of length for each member
    def random_get_length_size(self):
        length_size = {}
        size = 0
        for member in self.clique:
            length_size[member] = {}
            size = 0
            for length in self.parameters[member].keys():
                size += len(self.parameters[member][length])
                length_size[member][length] = size
        return length_size

    def create_random_range_queries(self):
        queries = {}
        for member in self.clique:
            queries[member] = {}
            for length in self.parameters[member].keys():
                queries[member][length] = []
                for count in range(len(self.parameters[member][length])):
                    queries[member][length] += [self.create_random_range_query(member, length, count)]
        return queries

    def create_random_range_query(self, member, length, count):
        return lambda df: self.answer_random_range_query(df, member, length, count)

    def answer_random_range_query(self, df, member, length, count):
        query_bounds = self.parameters[member][length][count]
        return df[df.eval('%d<=`%s` & `%s`<=%d' % (query_bounds[0], member, member, query_bounds[1]))]

    # Serves to the range query that generating queries non randomly
    #
    #
    #
    def generate_non_random_range_query_parameters(self):
        parameters = {}
        for member in self.clique:
            parameters[member] = {}
            query_parameter_list = self.non_random_range_partition(0, self.config[member]-1)
            query_parameter_dict = self.non_random_parameter_rearrange(query_parameter_list)
            parameters[member] = query_parameter_dict
        return parameters

    # This function returns the list that contains the query parameters
    def non_random_range_partition(self, lowerbound, upperbound):
        if lowerbound == upperbound:
            return [[lowerbound, upperbound]]
        else:
            partition_list = [[lowerbound, upperbound]]
            cut = int((lowerbound + upperbound) / 2)
            partition_list += self.non_random_range_partition(lowerbound, cut)
            partition_list += self.non_random_range_partition((cut + 1), upperbound)
            return partition_list

    def non_random_parameter_rearrange(self, query_list):
        length_list = []
        for pair in query_list:
            number = (pair[1] - pair[0] + 1)
            if number not in length_list:
                length_list.append(number)
        length_list.sort()
        query_dict = {}
        for length in length_list:
            query_dict[length] = []
            for pair in query_list:
                if (pair[1] - pair[0] + 1) == length:
                    query_dict[length] += [pair]
        return query_dict

    def non_random_get_length_size(self):
        length_size = {}
        size = 0
        for member in self.clique:
            length_size[member] = {}
            size = 0
            for length in self.parameters[member].keys():
                size += len(self.parameters[member][length])
                length_size[member][length] = size
        return length_size

    def create_non_random_range_queries(self):
        queries = {}
        for member in self.clique:
            queries[member] = {}
            for length in self.parameters[member].keys():
                queries[member][length] = []
                for count in range(len(self.parameters[member][length])):
                    queries[member][length] += [self.create_non_random_query(member, length, count)]
        return queries

    def create_non_random_query(self, member, length, count):
        return lambda df: self.answer_non_random_query(df, member, length, count)

    def answer_non_random_query(self, df, member, length, count):
        query_bounds = self.parameters[member][length][count]
        return df[df.eval('%d<=`%s` & `%s`<=%d' % (query_bounds[0], member, member, query_bounds[1]))]

    # Serves to the linear query
    #
    #
    #
    def generate_linear_query_parameters(self, query_size):
        parameters = {}
        for member in self.clique:
            np.random.seed(114514)
            query_parameter = np.random.rand(query_size, self.config[member])
            parameters[member] = query_parameter
        return parameters

    def create_linear_queries(self):
        queries = {}
        for member in self.clique:
            queries[member] = []
            for count in range(self.query_size):
                queries[member] += [self.create_linear_query(member, count)]
        return queries

    def create_linear_query(self, member, count):
        return lambda df: self.answer_linear_query(df, member, count)

    def answer_linear_query(self, df, member, count):
        query_parameter = self.parameters[member][count]
        answer = 0
        for attribute in range(self.config[member]):
            answer += len(df[df.eval('%d==%s' % (attribute, member))]) * query_parameter[attribute]
        return answer

    # auxiliary functions:
    def store_query_info(self, logger=None):
        logger.info('Query instance consist of clique: %s' % self.clique)
        for member in self.clique:
            if self.query_type == 'linear query':
                logger.info('linear query parameters are: %s' % self.parameters[member])
            elif self.query_type == 'range query':
                logger.info('range query_instance.length_size[%s]: %s' % (member, self.length_size[member]))
                for length in self.parameters[member].keys():
                    logger.info('for range query with length %d:' % length)
                    logger.info('range query parameters are with bounds: %s' % self.parameters[member][length])
        return

    def answer_queries(self, dataset, member):
        answer = []
        if self.query_type == 'linear query':
            for query in self.queries[member]:
                answer += [query(dataset)]
        elif self.query_type == 'range query':
            for length in self.queries[member]:
                for query in self.queries[member][length]:
                    answer += [len(query(dataset))]
        return answer

    def print_answer(self, answer, member):
        if self.query_type == 'linear query':
            print(answer)
        elif self.query_type == 'range query':
            head = 0
            tail = 0
            for length in self.queries[member].keys():
                tail = self.length_size[member][length]
                print('Range size %d, answer:' % length)
                print(answer[head: tail])
                head = self.length_size[member][length]
        return -1
