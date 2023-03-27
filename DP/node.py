import numpy as np
import pandas as pd
import math


class Node:
    def __init__(self, index, keys, sj=1):
        self.df = pd.DataFrame(columns=keys)
        self.delete_df = pd.DataFrame(columns=keys)
        self.index = index
        # self.sj refers to the end location of t when the node has been created
        self.sj = sj
        self.height = 0
        temp_index = index
        # Get the height
        if self.index == 0:
            return
        for self.height in range(0, int(math.log(self.index, 2) + 1)):
            if temp_index % 2 == 0:
                temp_index /= 2
            else:
                break
        # Case for the leaves
        if self.index % 2 != 0:
            self.right_child = None
            self.left_child = None
            if self.index % 4 == 1:
                self.right_ancestor_index = self.index + 1
                self.left_ancestor_index = None
            elif self.index % 4 == 3:
                self.right_ancestor_index = None
                self.left_ancestor_index = self.index - 1
        # Cases for the internal nodes
        elif temp_index % 4 == 1:
            self.right_child = self.index + 2 ** (self.height - 1)
            self.left_child = self.index - 2 ** (self.height - 1)
            self.right_ancestor_index = self.index + (2 ** self.height)
            self.left_ancestor_index = None
        elif temp_index % 4 == 3:
            self.right_child = self.index + 2 ** (self.height - 1)
            self.left_child = self.index - 2 ** (self.height - 1)
            self.right_ancestor_index = None
            self.left_ancestor_index = self.index - (2 ** self.height)

    def add_items(self, items, delete_df=False):
        if not delete_df:
            self.df = pd.concat([self.df, items], ignore_index=True)
        else:
            self.delete_df = pd.concat([self.delete_df, items], ignore_index=True)

    def add_item(self, item):
        self.df.loc[(len(self.df))] = item
        # self.df = self.df.append(item, ignore_index=True)

    def delete_item(self, item):
        for row in self.df.itertuples():
            if row.x == item.x and row.y == item.y:
                self.df.drop(row)
                break
        return

    def get_right_ancestor_index(self):
        return self.right_ancestor_index

    def get_left_ancestor_index(self):
        return self.left_ancestor_index

    def get_right_child(self):
        return self.right_child

    def get_left_child(self):
        return self.left_child

    def __repr__(self):
        return '\n-------- Node index: %s --------\n' \
               'height: %s, sj: %s, \nright_ancestor_index: %s, left_ancestor_index: %s, ' \
               'right_child: %s, left_child: %s, \ndf size: %s, delete_df size: %s\n' \
               '-----------------------------------' % (self.index,
                                                        self.height, self.sj,
                                                        self.right_ancestor_index,
                                                        self.left_ancestor_index,
                                                        self.right_child,
                                                        self.left_child, len(self.df),
                                                        len(self.delete_df))
        # return '\n-------- Node index: %s --------\n' \
        #        'height: %s, sj: %s, right_ancestor_index: %s, left_ancestor_index: %s, ' \
        #        'right_child: %s, left_child: %s, \ndf size: %s, delete_df size: %s\ndf: %s, \ndelete_df: %s' \
        #        '-----------------------------------' % (self.index,
        #                                                 self.height, self.sj,
        #                                                 self.right_ancestor_index,
        #                                                 self.left_ancestor_index,
        #                                                 self.right_child,
        #                                                 self.left_child, len(self.df),
        #                                                 len(self.delete_df), self.df, self.delete_df)
