import pandas as pd
import numpy as np
import configparser
import pickle

from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod

class AbstractPreprocessor(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def process(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode_series(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode_series(self, *args, **kwargs):
        pass

class Preprocessor(AbstractPreprocessor):
    
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
        
        # read in the config file
        self._cparser = configparser.ConfigParser()
        self._cparser.read(path_to_config)
        
        # grab all markers from the parser, which we want to have in our dataframe
        self._valid_markers = self._cparser.sections()

        if 'markerconfig_metainfo' in self._valid_markers:
            self._valid_markers.remove('markerconfig_metainfo')

        # get a hold of all the layers we have
        for marker in self._valid_markers:
            if 'layerName' in self._cparser[marker]:
                self._valid_layers.add(self._cparser[marker]['layerName'])

        # read in the data and store it into a dataframe
        self._df = pd.read_csv(path_to_data, delimiter=csv_delimiter, usecols=self._valid_markers)

        grouped_dfs = self._group()
        consistant_dfs = self._enforce_timeinterval_consistency(grouped_dfs)
        self._df = pd.concat(consistant_dfs).reset_index(drop=True)
        
        self._build_label_encoders()
        
    def _group(self) -> [pd.DataFrame]:
        '''
    
        returns :   A list of DataFrames grouped by the given attribute, if no attribute was supplied inside the
                    cparser, then the received DataFrame is returned without grouping, as a single element list.
        '''
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


    
    def _enforce_timeinterval_consistency(self, dfs : [pd.DataFrame]) -> [pd.DataFrame]:
    
        if 'markerconfig_metainfo' not in self._cparser or 'dateinfo' not in self._cparser['markerconfig_metainfo']:
            return dfs

        dateid, time1, time2 = eval(self._cparser['markerconfig_metainfo']['dateinfo'])
        timedelta = abs(pd.Timestamp(time1) - pd.Timestamp(time2))

        for i in range(len(dfs)):
            dfs[i] = self._make_df_timeinterval_consistent(dfs[i], timedelta, dateid)


        return dfs
        
    def _make_df_timeinterval_consistent(self, df : pd.DataFrame, timedelta : pd.Timedelta, dateid : str) -> pd.DataFrame:
        '''
        df :        The dataframe we want to enforce time interval consistency on
        timedelta : Timedelta object, the maximum time delta, that two observations are allowed away from each other.
        returns :   A time inteval consistent DataFrame. The new DataFrame may contain empty rows (filler rows are added
                    when two observations are too far away from each other), or fewer rows than before (when observations are
                    to close to each other, say all on the same day, we have to drop all but one).
        '''
        if dateid not in df.columns:
            raise KeyError(f"dateid-key {dateid}, which was specified inside the config.ini was not in the data.")

        # 1) remove duplicate values
        df = df.drop_duplicates(subset=[dateid])


        # 2) map the dates to pd.Timestamp objects
        try:
            df[dateid] = df[dateid].map(lambda x: pd.Timestamp(x))
        except Exception as e:
            print('There has been a problem parsing the dates in your data!'\
                   + str(e)\
                  +'Please let your dates conform to the ISO 8601 standart YYYY-MM-DD hh:mm:ss')
            return pd.DataFrame()


        # sort the dataframe by dates
        df = df.sort_values(by=[dateid]).reset_index(drop=True)

        # 3) Allocate new rows inside a list. Fill in empty rows, in case we have a temporal gap
        new_rows = []
        dates = df[dateid]

        # always add the first date. In the following loop, we will add subsequent dates,
        # aswell as filler rows.
        new_rows.append(df.iloc[0].to_dict())

        for i in range(1, len(dates)):
            current_date = dates.iloc[i-1]
            next_date = dates.iloc[i]

            if timedelta < next_date - current_date:

                cur_dict = df.iloc[i-1].to_dict()

                # add some filler rows
                periods = 2 + int(abs(next_date - current_date) / timedelta)

                times = pd.date_range(start=current_date, end=next_date, periods=periods)

                rows = [cur_dict.copy() for _ in times[1:-1]]

                for j, date in enumerate(times[1:-1]):
                    rows[j][dateid] = date

                new_rows = new_rows + rows


            # append next row to list
            new_rows.append(df.iloc[i].to_dict())


        return pd.DataFrame(new_rows, columns=df.columns)


    def _build_label_encoders(self):

        for marker in self._valid_markers:
            dtype = self._cparser[marker]['dtype']

            if 'linspace' in dtype:
                try:
                    start, end, number_bins = eval(dtype.replace('linspace', ''))
                    start = float(start)
                    end = float(end)
                    number_bins = int(number_bins)

                    bins = np.linspace(start, end, number_bins)

                    # transform section of df with the help of bins!
                    self._df[marker] = self._df[marker].map(lambda x: np.digitize(x, bins))
                    self._bin_dict[marker] = bins
                
                except:
                    print(f"dtype of marker {marker} was parsed as {dtype}, but we encountered an error.")
                    raise

            # construct label encoder, fit label encoder, transform column
            tmp_encoder = LabelEncoder()
            tmp_encoder.fit(self._df[marker])
            self._df[marker] = tmp_encoder.transform(self._df[marker])

            # save label encoder
            self._label_encoders[marker] = tmp_encoder

    def encode_series(self, marker, series : pd.Series):
        
        dtype = self._cparser[marker]['dtype']

        if 'linspace' in dtype:
            try:
                start, end, number_bins = eval(dtype.replace('linspace', ''))
                start = float(start)
                end = float(end)
                number_bins = int(number_bins)

                bins = np.linspace(start, end, number_bins)

                # transform section of df with the help of bins!
                result = series.map(lambda x: np.digitize(x, bins))

            except:
                print(f"dtype of marker {marker} was parsed as {dtype}, but we encountered an error.")
                raise
        else:
            result = series

        # its still possible, that we get previously unseen labels, which we cannot encode.
        return self._label_encoders[marker].transform(result)

    
    def decode_series(self, marker, series : pd.Series):

        dtype = self._cparser[marker]['dtype']
        try:
            result = self._label_encoders[marker].inverse_transform(series)
        except:
            if self._debug:
                print(f"Inverse transform was faulty for series {series} and marker {marker}")
            raise

        if 'linspace' in dtype or 'date_range' in dtype:    
            result = np.array(list(map(lambda x: self._bin_dict[marker][x - 1], result)))

        return result
    
    def to_pickle(self, filename : str):
        
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        with open(filename, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    def group_df(self, df : pd.DataFrame, make_time_interval_consistant=False) -> [pd.DataFrame]:

        grouping_handle = self.__check_for_groupid()

        if grouping_handle is not None:
            tmpdf = df.groupby(by=grouping_handle)
            dfs = [tmpdf.get_group(group) for group in tmpdf.groups]
        else:
            dfs = [df]

        if make_time_interval_consistant:
            return self._enforce_timeinterval_consistency(dfs)
        else:
            return dfs

    def __check_for_groupid(self):
        if 'markerconfig_metainfo' not in self._cparser or 'groupby' not in self._cparser['markerconfig_metainfo']:
            return None
        return self._cparser["markerconfig_metainfo"]["groupby"]
