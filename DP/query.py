import random


class Query:
    def __init__(self, config, number=5, query_size=10):
        self.clique = random.sample(config.keys(), number)
        self.config = config
        self.queries = self.create_queries(query_size)

    def create_queries(self, query_size):
        queries = {}
        for member in self.clique:
            queries[member] = []
            for count in range(query_size):
                queries[member] += [self.create_query(member)]
        return queries

    def answer_query_instance(self, df, member):
        query_bounds = sorted(random.sample(range(self.config[member]), 2))
        return df[df.eval('%d<`%s` & `%s`<%d' % (query_bounds[0], member, member, query_bounds[1]))]

    def create_query(self, member):
        return lambda df: self.answer_query_instance(df, member)
