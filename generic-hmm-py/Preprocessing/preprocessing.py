import pandas as pd
import numpy as np
import configparser
import pickle

from sklearn.preprocessing import LabelEncoder
from source.pyth.Preprocessing.Interface.abstract_preprocessor import AbstractPreprocessor


class Preprocessor(AbstractPreprocessor):
    """
    A simple preprocessor implementation.
    """
    def __init__(self, debug=False):

        super().__init__()

        self._label_encoders = {}
        self._bin_dict = {}
        self._valid_markers = set()
        self._valid_layers = set()
        self._cparser = None
        self._df = None
        self._debug = debug

    def process(self, path_to_config : str, path_to_data : str, csv_delimiter = '\t'):
        """
        :param path_to_config: Path to the configuration file
        :param path_to_data: Path to the csv data file
        :param csv_delimiter: delimiter used in your csv file., defaults to '\t'

        :return: returns None, processes the dataframes
        :rtype: None
        """
        # read in the config file
        return None

    def _group(self) -> [pd.DataFrame]:
        """

        :return: returns list of grouped dataframes
        :rtype: List[pandas.DataFrame]
        """
        # check if we have metainfo, if we dont,
        # we have to treat the data as one big experiment
        if 'markerconfig_metainfo' not in self._cparser or 'groupby' not in self._cparser['markerconfig_metainfo']:
            return [self._df]

        # load our key, witch which we are grouping the data.
        key = self._cparser['markerconfig_metainfo']['groupby']

        if key not in self._df.columns:
            raise KeyError(f"The groupby-key '{key}' specified inside the config file isn't found in the data.")

        tmpdf = self._df.groupby(key)
        return [tmpdf.get_group(group) for group in tmpdf.groups]
