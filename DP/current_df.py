import pandas as pd


class CurrentDf:
    def __init__(self, keys):
        self.data = pd.DataFrame(columns=keys)

    def current_df_update(self, item, index):
        if item.loc[index, 'update'] == 1:
            self.add_item(item)
        elif item.loc[index, 'update'] == -1:
            self.delete_item(item)
        return

    def add_item(self, item):
        self.data = pd.concat([self.data, item.drop(columns=['update'])], ignore_index=True)

    def delete_item(self, item):
        cur_item = item.drop(columns=['update'])
        self.data = pd.concat([self.data, cur_item, cur_item]).drop_duplicates(keep=False)

    # This function only serves to cur_deletion_df
    def add_deletion_item(self, item, index):
        if item.loc[index, 'update'] == -1:
            self.add_item(item)
        return

    def renew(self):
        self.data = self.data.drop(index=self.data.index)
        return
