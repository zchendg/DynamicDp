import os
import pandas as pd
import json
import my_logging
import argparse
import auxiliary
from datetime import datetime
from dynamic_tree import DynamicTree
from query import Query


def range_partition(lowerbound, upperbound):
    if lowerbound == upperbound:
        return [[lowerbound, upperbound]]
    else:
        partition_list = [[lowerbound, upperbound]]
        cut = int((lowerbound + upperbound)/2)
        partition_list += range_partition(lowerbound, cut)
        partition_list += range_partition((cut + 1), upperbound)
        return partition_list

def rearrange(list):
    length_list = []
    for pair in list:
        number = (pair[1] - pair[0] + 1)
        if number not in length_list:
            length_list.append(number)
    length_list.sort()
    query_dict = {}
    for length in length_list:
        query_dict[length] = []
        for pair in list:
            if (pair[1] - pair[0] + 1) == length:
                query_dict[length] += [pair]
    return query_dict

def main():
    partition_list = range_partition(0,17)
    print(partition_list)
    partition_list = rearrange(partition_list)
    print(partition_list)
    config = json.load(open('./data/adult-domain.json'))
    query_instance = Query(config)


if __name__ == '__main__':
    main()
