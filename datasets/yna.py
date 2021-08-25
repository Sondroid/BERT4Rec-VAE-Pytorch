from .base import AbstractDataset

import pandas as pd

class YNADataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'yna'

    @classmethod
    def url(cls):
        pass

    @classmethod
    def zip_file_content_is_folder(cls):
        pass

    @classmethod
    def all_raw_file_names(cls):
        pass

    def load_clicks_df(self):
        folder_path = self._get_rawdata_folder_path()

        train_file_path = folder_path.joinpath(self.dataset_datetime + '_training.txt')
        val_file_path = folder_path.joinpath(self.dataset_datetime + '_validation.txt')
        test_file_path = folder_path.joinpath(self.dataset_datetime + '_test.txt')

        train = pd.read_csv(train_file_path, sep=',', header=0)
        val = pd.read_csv(val_file_path, sep=',', header=0)
        test = pd.read_csv(test_file_path, sep=',', header=0)

        train.columns = ['uid', 'timestamp', 'sid']
        val.columns = ['uid', 'timestamp', 'sid']
        test.columns = ['uid', 'timestamp', 'sid']

        return train, val, test