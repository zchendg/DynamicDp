import random


class Query:
    def __init__(self, config, random_query=False, column_number=2, query_size=10, logger=None):
        # self.clique = random.sample(config.keys(), column_number)
        self.clique = ['age']
        self.config = config
        if random_query:
            self.parameters = self.generate_query_parameters(query_size)
            self.length_size = self.get_length_size()
            self.queries = self.create_queries()
        else:
            self.parameters = self.generate_query_parameters_non_randomly()
            self.length_size = self.get_length_size()
            self.queries = self.create_queries()
        self.store_query_info(logger)

    def generate_query_parameters_non_randomly(self):
        parameters = {}
        for member in self.clique:
            parameters[member] = {}
            query_parameter_list = self.range_partition(0, self.config[member]-1)
            query_parameter_dict = self.rearrange(query_parameter_list)
            parameters[member] = query_parameter_dict
        return parameters

    # Store size of length for each member
    def get_length_size(self):
        length_size = {}
        size = 0
        for member in self.clique:
            length_size[member] = {}
            size = 0
            for length in self.parameters[member].keys():
                size += len(self.parameters[member][length])
                length_size[member][length] = size
        return length_size

    def create_queries(self):
        queries = {}
        for member in self.clique:
            queries[member] = {}
            for length in self.parameters[member].keys():
                queries[member][length] = []
                for count in range(len(self.parameters[member][length])):
                    queries[member][length] += [self.create_query(member, length, count)]
        return queries

    # This function returns the list that contains the query parameters
    def range_partition(self, lowerbound, upperbound):
        if lowerbound == upperbound:
            return [[lowerbound, upperbound]]
        else:
            partition_list = [[lowerbound, upperbound]]
            cut = int((lowerbound + upperbound) / 2)
            partition_list += self.range_partition(lowerbound, cut)
            partition_list += self.range_partition((cut + 1), upperbound)
            return partition_list

    # This function returns the dict that contains the length to the query parameters
    def rearrange(self, query_list):
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

    # create_queries is for random querying, above serves to the formal query
    def create_queries_randomly(self, query_size):
        queries = {}
        for member in self.clique:
            queries[member] = []
            for count in range(query_size):
                queries[member] += [self.create_query(member, count)]
        return queries

    def create_query(self, member, length, count):
        return lambda df: self.answer_query_instance(df, member, length, count)

    def generate_query_parameters(self, query_size, threshold=1):
        parameters = {}
        for member in self.clique:
            parameters[member] = {}
            query_parameter_list = []
            for count in range(query_size):
                query_bounds = query_bounds = sorted(random.sample(range(self.config[member]), 2))
                # query_bounds = self.generate_query_bound(member, int((1/3)*self.config[member]))
                query_parameter_list += [query_bounds]
            query_parameter_dict = self.rearrange(query_parameter_list)
            parameters[member] = query_parameter_dict
        return parameters

    def answer_query_instance(self, df, member, length, count):
        query_bounds = self.parameters[member][length][count]
        return df[df.eval('%d<=`%s` & `%s`<=%d' % (query_bounds[0], member, member, query_bounds[1]))]

    def answer_queries(self, dataset, member):
        answer = []
        for length in self.queries[member]:
            for query in self.queries[member][length]:
                answer += [len(query(dataset))]
        return answer

    def output_answer(self, answer, member):
        head = 0
        tail = 0
        for length in self.queries[member].keys():
            tail = self.length_size[member][length]
            print('Range size %d, answer:' % length)
            print(answer[head: tail])
            head = self.length_size[member][length]
        return -1

    def store_query_info(self, logger=None):
        logger.info('Query instance consist of clique: %s' % self.clique)
        for member in self.clique:
            logger.info('query_instance.length_size[%s]: %s' % (member, self.length_size[member]))
            for length in self.parameters[member].keys():
                logger.info('for query with length %d:' % length)
                logger.info('query parameters is with bounds: %s' % self.parameters[member][length])
        return

    def generate_query_bound(self, member, threshold=1):
        query_bounds = sorted(random.sample(range(self.config[member]), 2))
        counter = 0
        while counter < 50:
            if (query_bounds[1]-query_bounds[0]+1) >= threshold:
                return query_bounds
            else:
                query_bounds = sorted(random.sample(range(self.config[member]), 2))
                counter+1
        return query_bounds
