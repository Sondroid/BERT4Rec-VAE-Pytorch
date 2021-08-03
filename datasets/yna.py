from .base import AbstractDataset

import pandas as pd

from datetime import date


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

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('20210712 1210.csv')
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['uid', 'timestamp', 'sid']
        return df


