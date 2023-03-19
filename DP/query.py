import random


class Query:
    def __init__(self, config, column_number=5, query_size=10):
        # self.clique = random.sample(config.keys(), column_number)
        self.clique = ['age']
        self.config = config
        self.parameters = self.generate_query_parameters(query_size)
        self.queries = self.create_queries(query_size)

    def answer_query_instance(self, df, member, count):
        query_bounds = self.parameters[member][count]
        return df[df.eval('%d<`%s` & `%s`<%d' % (query_bounds[0], member, member, query_bounds[1]))]

    def create_queries(self, query_size):
        queries = {}
        for member in self.clique:
            queries[member] = []
            for count in range(query_size):
                queries[member] += [self.create_query(member, count)]
        return queries

    def create_query(self, member, count):
        return lambda df: self.answer_query_instance(df, member, count)

    def generate_query_parameters(self, query_size):
        parameters = {}
        for member in self.clique:
            parameters[member] = []
            for count in range(query_size):
                query_bounds = sorted(random.sample(range(self.config[member]), 2))
                parameters[member] += [query_bounds]
        return parameters
