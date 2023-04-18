import sys
import pandas as pd
import numpy as np
import json
import logging
import os
import argparse
import auxiliary
from node import Node
from approximation_instance import ApproximationInstance
from insertion_mechanism import Insertion_Mechanism
import auxiliary
import auxiliary1
from mbi import Domain, Dataset


class Insertion_Deletion_Mechanism:
    def __init__(self, config, query_instance=None):
        self.insertion_tree = Insertion_Mechanism(config, query_instance)
        self.deletion_tree = Insertion_Mechanism(config, query_instance)
        self.query_instance = query_instance
        self.answer_ground_truth = {}
        self.answer_mechanism = {}
        self.initialize_answer()

    def testing(self, ipp_instance, epsilon=1, delta=0, beta=0.05, iteration=500, logger=None):
        self.insertion_tree.testing(ipp_instance, epsilon, delta, beta, iteration, logger)
        self.deletion_tree.testing(ipp_instance, epsilon, delta, beta, iteration, logger)
        self.compute_difference(ipp_instance)
        self.store_answer(ipp_instance, logger)

    def compute_difference(self, ipp_instance):
        members = self.query_instance.queries.keys()
        indexes = range(1, len(ipp_instance.get_segment()) - 1)
        for member in members:
            for index in indexes:
                self.answer_ground_truth[member][index] = self.insertion_tree.answer_ground_truth[member][index] - \
                                                          self.deletion_tree.answer_ground_truth[member][index]
                self.answer_mechanism[member][index] = self.insertion_tree.answer_mechanism[member][index] - \
                                                       self.deletion_tree.answer_mechanism[member][index]
        return

    def store_answer(self, ipp_instance, logger=None):
        logger.info('Insertion and Deletion Mechanism:')
        clique = self.query_instance.queries.keys()
        indexes = range(1, len(ipp_instance.get_segment()) - 1)
        for member in clique:
            logger.info('Member %s' % member)
            for index in indexes:
                logger.info('For ground truth, queries on node %d gives answer %s' % (
                    index, self.answer_ground_truth[member][index]))
                logger.info('For insertion-only mechanism, queries on node %d gives answer %s' % (
                    index, self.answer_mechanism[member][index]))
        logger.info('Insertion and Deletion Mechanism finished')

    def initialize_answer(self):
        for member in self.query_instance.queries.keys():
            self.answer_ground_truth[member] = {}
            self.answer_mechanism[member] = {}
        return
