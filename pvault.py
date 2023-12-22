import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd

from typing import List
from tqdm import tqdm
from preprocessing import AbstractPreprocessor, Preprocessor
from abc import ABC, abstractmethod

class AbstractFeatureExtractor(ABC):

    def __init__(self, prep : AbstractPreprocessor, *args, **kwargs):
        pass

    @abstractmethod
    def extract_probabilities(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_layerinfo_from_layers(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_dateid_if_present(self, *args, **kwargs):
        pass

class ProbabilityVault(AbstractFeatureExtractor):
    """
    We want to archieve a structure that helps us easily query for probabilities,
    whilst not having to rely on huge matrices for storage.

    _initial_state_probs : These are mappings from names of markers to distributions over states resembling
    the starting state distribution.

    _initial_state_probs = {'marker1' :
                               { 0 : 0.2, 1 : 0.3 ... }
                            'marker2' :
                               { 0 : 0.3, 1 : 0.1 ... }
                           ...

                           }

    _state_transition_probs : Here, we have a map from marker to a 2-d dictionary. This
    allows for quick access, serializability and relatively small memory space, since we expect the
    state transitions to be sparse for big state spaces

    _state_transition_probs = {'marker1' :
                                  { 0 :
                                      {0 : 0.2, 1 : 0.8}},
                                  { 1 :
                                      {1 : 0.4, 3 : 0.4, 5 : 0.2}}, ... ,
                               'marker2' :
                                  { 0 :
                                      {0 : 1.0}},
                                  { 1 :
                                      {1 : 0.1, 2 : 0.1, 3 : 0.9}}, ... ,

                               ... }

    _observation_probs : Here, we have a map from marker to another marker to a 2-d dictionary.
    Although weird looking, we avoid enormous memory allocation since we do not have to allocate M^2 * S^2 cells, where
    N is the mean number of Markers and S the mean number of States.

    _observation_probs = {'marker1' :
                                  {'marker2' :
                                      { (marker1, state1) 0 :
                                          { (marker2, state1) 0 : 0.2, 1 : 0.8}},
                                      { 1 :
                                          {1 : 0.4, 3 : 0.4, 5 : 0.2}}, ... ,
                                   'marker3' :
                                      { (marker1, state1) 0 :
                                          { (marker3, state1) 0 : 1.0}},
                                      { 1 :
                                          {1 : 0.1, 2 : 0.1, 3 : 0.9}}, ... ,

                                   ... }
                           'marker2' :
                                  {'marker1' :
                                      { 0 :
                                          {0 : 0.2, 1 : 0.8}},
                                      { 1 :
                                          {1 : 0.4, 3 : 0.4, 5 : 0.2}}, ... ,
                                   'marker3' :
                                      { 0 :
                                          {0 : 1.0}},
                                      { 1 :
                                          {1 : 0.1, 2 : 0.1, 3 : 0.9}}, ... ,

                                   ... }

                           ... }

    """

    def __init__(self, prep: Preprocessor, debug=False):
        """constructor method"""
        super().__init__(prep)

        self._initial_state_probs = dict()
        self._state_transition_probs = dict()
        self._observation_probs = dict()

        self._prep = prep
        self._label_encoders = prep._label_encoders
        self._cparser = prep._cparser
        self._df = prep._df
        self._markers = prep._valid_markers
        self._layers = prep._valid_layers
        self._initialized = False
        self._debug = debug

    def extract_probabilities(self):

        starting_time = time.time()

        self._setup_prob_dicts()

        self._extract_counts()

        self._normalize_counts()

        duration_sec = time.time() - starting_time

        hrs  = duration_sec // 3600
        mins = (duration_sec // 60) % 60
        secs = duration_sec % 60

        print(f'Extraction of probabilities successful. Elapsed time: {hrs} hours, {mins} minutes and {secs} seconds')

        self._initialized = True

    @staticmethod
    def from_pickle(path_to_pickle : str):

        if not os.path.isfile(path_to_pickle) or not path_to_pickle.endswith('.pkl'):
            raise FileNotFoundError("The path specified is either invalid or doesn't point to a .pkl file.")

        with open(path_to_pickle, 'rb') as file:
            pvault = pickle.load(file)

        return pvault

    def _normalize_counts(self):

        """
        The goal of this function is to normalize the counts of observations/state transitions/initial states, which
        were collected in the _exctract_counts() function.

        We simply iterate over the dictionaries, summing along the respective axes before normalizing the counts.
        This results in a probability distribution over the respective states of markers.

        """

        print('Calculating initial state probabilities.')
        for marker, counts in tqdm(self._initial_state_probs.items()):

            _sum = sum(counts.values())

            if _sum == 0:
                continue

            normalization_factor = 1.0 / _sum
            self._initial_state_probs[marker] = {k : normalization_factor*v for k,v in counts.items()}

        print('Calculating state transition probabilities.')
        for marker, state_dict in tqdm(self._state_transition_probs.items()):

            for state, counts in state_dict.items():

                _sum = sum(counts.values())

                # we have to check for empty state transition dictionaries
                # (absorbing states at the end of observation sequences)
                if _sum == 0:
                    clear_state_name = self._prep.decode_series(series=pd.Series(state), marker=marker)[0]
                    warnings.warn(f"We have encountered a possible absorbing state {clear_state_name} "
                                  f"for marker {marker}. The state was observed in the training dataset, but didn't have any"
                                  f"following states. The transition probability to the state itself will be set to 1.")
                    self._state_transition_probs[marker][state] = {state : 1.0}
                    continue


                normalization_factor = 1.0 / _sum
                self._state_transition_probs[marker][state] = {k : normalization_factor*v for k,v in counts.items()}

        print('Calculating observation probabilities.')
        for marker1, m1_dict in tqdm(self._observation_probs.items()):
            for marker2, s1_dict in m1_dict.items():
                for state, counts in s1_dict.items():

                    _sum = sum(counts.values())

                    if _sum == 0:
                        raise ValueError(f"We have encountered an unexpected error. The sum over the "
                                      f"observation-counts for marker1 {marker1}, marker2 {marker2}, state1 {state} was 0.")

                    normalization_factor = 1.0 / _sum
                    self._observation_probs[marker1][marker2][state] = {k : normalization_factor*v for k,v in counts.items()}

        return


    def _extract_counts(self):

        """
        The goal of this function is to make the first step towards calculating the actual
        transition/observation/initial_state probabilities.

        To calculate the probabilities, we first have to count occurences of observations or state transitions.
        This is done in this function. We iterate over the dataframes, which are grouped by experiment, and
        increment a counter inside a dictionary.

        """

        if 'markerconfig_metainfo' in self._cparser and 'groupby' in self._cparser['markerconfig_metainfo']:
            tmpdf = self._df.groupby(self._cparser['markerconfig_metainfo']['groupby'])
            experiments = [tmpdf.get_group(group) for group in tmpdf.groups]
        else:
            experiments = [self._df]



        print('Extracting counts. This may take some time.')
        for df in tqdm(experiments):
            if df.empty:
                continue

            # assert that index is reset
            df = df.reset_index(drop=True)

            # get the first row, update init_state_probs, and observation_probs
            first_row = df.iloc[0]

            # keep track of the last row, in order to increment the state transitions.
            last_row = first_row.to_dict()

            # loop over markers in the first row, increment the initial state probability entry.
            # (later on we will normalize over all states of a marker)
            for marker1 in self._markers:

                current_state = first_row[marker1]

                self._incr_init_state_probs_count(marker1, current_state)

                for marker2 in self._markers:
                    if marker1 == marker2:
                        continue

                    other_state = first_row[marker2]

                    self._incr_obs_probs_count(marker1, marker2, current_state, other_state)


            """
            Now that we have extracted the initial state probabilities from the first row, we should
            continue extracting information from the subsequent rows. Here, we have to

            1) increment the state transition probability count by using the elements inside the last_row dictionary
            and the elements inside the current row

            2) increment the observation probability count by comparing the elements in the current row with each other.

            3) Once we are done in the current row, set the last_row dictionary to be the current row

            """

            for i in range(1, df.shape[0]):

                current_row = df.iloc[i]

                for marker1 in self._markers:

                    last_state = last_row[marker1]
                    current_state = current_row[marker1]

                    # 1) increment the state transition probability count

                    self._incr_state_trans_probs_count(marker1, last_state, current_state)


                    for marker2 in self._markers:

                        if marker1 == marker2:
                            continue

                        other_state = current_row[marker2]

                        # 2) increment the observation probability count

                        self._incr_obs_probs_count(marker1, marker2, current_state, other_state)

                # 3) set the last_row dictionary to be the current row
                last_row = current_row.to_dict()

        return

    def _setup_prob_dicts(self):
        """
        Constructing the probability maps, we take care to not preemtively assign empty placeholders
        for values which are calculated later on. This would completely negate the positive effect of
        constructing the maps this way, instead of simply allocating alot of memory for transition/observation matrices.

        Thus, any key or value which **may not exist** when we are finished computing the data, is not added in the beginning.
        """

        for marker1 in self._markers:

            marker1_num_states = self._label_encoders[marker1].classes_.shape[0]
            self._initial_state_probs[marker1] = {}
            self._state_transition_probs[marker1] = {i : {} for i in range(marker1_num_states)}
            self._observation_probs[marker1] = {}

            for marker2 in self._markers:

                if marker1 == marker2:
                    continue

                self._observation_probs[marker1][marker2] = {i : {} for i in range(marker1_num_states)}

        return

    def _incr_obs_probs_count(self, marker1, marker2, current_state, other_state):

        if marker1 == marker2:
            return

        if current_state not in self._observation_probs[marker1][marker2]:
            self._observation_probs[marker1][marker2][current_state] = {}

        incremented_value = self._observation_probs[marker1][marker2][current_state].get(other_state, 0) + 1
        self._observation_probs[marker1][marker2][current_state][other_state] = incremented_value

        return

    def _incr_state_trans_probs_count(self, marker, last_state, current_state):

        assert last_state in self._state_transition_probs[marker]

        incremented_value = self._state_transition_probs[marker][last_state].get(current_state, 0) + 1
        self._state_transition_probs[marker][last_state][current_state] = incremented_value

        return

    def _incr_init_state_probs_count(self, marker, current_state):

        assert marker in self._initial_state_probs

        incremented_value = self._initial_state_probs[marker].get(current_state, 0) + 1
        self._initial_state_probs[marker][current_state] = incremented_value

        return



    def to_pickle(self, path_to_pickle : str):

        with open(path_to_pickle, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)



    def get_initial_prob(self, marker, state):

        if not self._initialized:
            raise RuntimeError("ProbabilityVault object isn't initalized yet. Please call extract_probabilities first, before querying")

        if not marker in self._initial_state_probs:
            raise KeyError(f"Marker {marker} wasn't found inside the initial state probability map. Please check for any spelling mistakes.")

        return self._initial_state_probs[marker].get(state, 0.0)

    def get_transition_prob(self, marker, state1, state2):
        """
        marker: The Marker which the two states belong to
        state1:  current state
        state2:  other state
        """
        if not self._initialized:
            raise RuntimeError("ProbabilityVault object isn't initalized yet. Please call extract_probabilities first, before querying")

        if not marker in self._state_transition_probs:
            raise KeyError(f"Marker {marker} wasn't found inside the state transition probability map. Please check for any spelling mistakes.")

        if not state1 in self._state_transition_probs[marker]:
            raise KeyError(f"State {state1} wasn't found inside the state transition probability map under marker {marker}. Please check for any spelling mistakes.")

        return self._state_transition_probs[marker][state1].get(state2, 0.0)

    def get_observation_prob(self, marker1, marker2, hidden, observed):
        """
        marker1: The Marker which the hidden state state1 belongs to
        marker2: The Marker which is the observed state.
        state1:  Hidden State
        obs:     observed State
        """
        if not self._initialized:
            raise RuntimeError("ProbabilityVault object isn't initalized yet. Please call extract_probabilities first, before querying")

        if not marker1 in self._observation_probs:
            raise KeyError(f"Marker {marker1} wasn't found inside the observation probability map. Please check for any spelling mistakes.")

        if not marker2 in self._observation_probs[marker1]:
            raise KeyError(f"Marker {marker2} wasn't found inside the observation probability map under marker1 {marker1}. Please check for any spelling mistakes.")

        if not hidden in self._observation_probs[marker1][marker2]:
            raise KeyError(f"State {hidden} wasn't found inside the state observation probability map map under marker1 {marker1} -> marker2 {marker2}. Please check for any spelling mistakes.")


        return self._observation_probs[marker1][marker2][hidden].get(observed, 0.0)

    def get_isp_dict(self):

        return self._initial_state_probs

    def get_stp_dict(self):

        return self._state_transition_probs

    def get_op_dict(self):

        return self._observation_probs

    def get_layerinfo_from_layers(self, hidden_marker : str, layers : List[str]) -> dict:

        # remove any layers which have never appeared before
        for layer in layers:
            if layer not in self._layers:
                warnings.warn(f"layer {layer} was to be consulted for prediction, but was not found"
                 "in the list of valid layers. Please check for spelling.")
                layers.remove(layer)


        layerinfo = {layer : {'markers' : {}, 'weight' : None} for layer in layers}

        # get layer weights from configparser
        if 'markerconfig_metainfo' in self._cparser and 'layerinfo' in self._cparser["markerconfig_metainfo"]:
            layer_weight_map = eval(self._cparser["markerconfig_metainfo"]['layerinfo'])

            for key, value in layer_weight_map.items():
                if value < 0 or value > 1:
                    raise ValueError(f"value for key {key} was {value}. Please only give values in the interval [0,1].")
            if abs(sum(layer_weight_map.values()) - 1) >= 1e-10:
                raise ValueError(f"Sum over layer-weights was found to be {sum(layer_weight_map.values())}. Please let the layer-weights sum to 1.")

            for key,value in layer_weight_map.items():
                layerinfo[key]['weight'] = value

        else:
            length = len(layerinfo.keys())

            if length == 0:
                raise ValueError("length of layerinfo list was zero. There cannot be a prediction without prediction layers. Please specify.")

            value = 1 / len(layerinfo.keys())
            for key in layerinfo.keys():
                layerinfo[key]['weight'] = value

        # fill in layerinfo dict with information about markers
        for marker in self._markers:

            # we do not want to consult the hidden marker for predictions on itself.
            if marker == hidden_marker:
                continue

            if 'layerName' in self._cparser[marker] and self._cparser[marker]['layerName'] in layers:
                layerinfo[self._cparser[marker]['layerName']]['markers'][marker] = {}

                weight = float(self._cparser[marker]['layerSpecificWeight'].replace(",", ".")) if 'layerSpecificWeight' in self._cparser[marker] else 0.0
                layerinfo[self._cparser[marker]['layerName']]['markers'][marker]['weight'] = weight

        # normalize the weights of the markers inside a layer.
        # if the markers have no weights assigned to them, set all their weights equal

        for layername, d in layerinfo.items():

            weights_summed = sum([d['markers'][m]['weight'] for m in d['markers'].keys()])

            if weights_summed - 1 > 1e-10:
                warnings.warn(f"In layer {layername}, the sum of weights, here being {weights_summed}, exceeds 1.0. The weights will be normalized.")

            # 1) no weights are assigned
            elif weights_summed == 0:
                new_weight = 1.0 / len(d['markers'].keys())

                for m in d['markers'].keys():
                    layerinfo[layername]['markers'][m]['weight'] = new_weight


            elif weights_summed < 0:
                raise ValueError(f"sum of weights of layer {layername} was negative: {weights_summed}")
            # 2) the weights must be normalized
            else:
                normalization_factor = 1.0 / weights_summed

                for m in d['markers'].keys():
                    layerinfo[layername]['markers'][m]['weight'] *= normalization_factor

        # remove any layer which does not have any markers in it.
        remove_keys = []
        for layername, d in layerinfo.items():
            if len(d['markers'].keys()) == 0:
                remove_keys.append(layername)

        for rk in remove_keys:
            del layerinfo[rk]

        return layerinfo

    def get_dateid_if_present(self):

        if 'markerconfig_metainfo' in self._cparser and 'dateinfo' in self._cparser['markerconfig_metainfo']:
            dateid, _,_ = eval(self._cparser['markerconfig_metainfo']['dateinfo'])
            return dateid

        return None

    def num_states_for_marker(self, marker):

        return self._label_encoders[marker].classes_.shape[0]

    def encode(self, marker, series):

        return self._prep.encode_series(marker, series)

    def decode(self, marker, series):

        return self._prep.decode_series(marker, series)

    def get_cparser(self):
        return self._cparser

    def transition_matrix(self, marker):
        num_states = self.num_states_for_marker(marker)

        ret = np.zeros(shape=(num_states, num_states))

        for state, transitions in self._state_transition_probs[marker].items():

            for other, prob in transitions.items():
                ret[int(state), int(other)] = float(prob)

        if self._debug:
            # validation of stochastic contrains
            for i in range(num_states):
                _sum = sum(ret[i,:])
                if _sum <= 0 or abs(_sum - 1) > 1e-10:
                    raise ValueError(f'Sum over columns of stochastic matrix was out of valid \
                    bounds for marker {marker} for row {i} with value {_sum}')
        return ret

    def initial_state_distribution(self, marker):

        num_states = self.num_states_for_marker(marker)

        ret = np.zeros(num_states)

        for state, prob in self._initial_state_probs[marker].items():
            ret[int(state)] = float(prob)

        if self._debug and sum(ret) < 0 or sum(ret) - 1 > 1e-10:
            raise ValueError(f'Sum over initial state probabilities was out of valid\
             bounds for marker {marker} with value {sum(ret)}')
        return ret

    def emission_matrix(self, marker1, marker2):
        marker1_states = self.num_states_for_marker(marker1)
        marker2_states = self.num_states_for_marker(marker2)


        ret = np.zeros(shape=(marker1_states, marker2_states))

        for state, distr in self._observation_probs[marker1][marker2].items():

            for other, p in distr.items():
                ret[int(state), int(other)] = float(p)

        if self._debug:

            for i in range(marker1_states):

                _sum = sum(ret[i, :])

                if _sum <= 0 or abs(_sum - 1) > 1e-10:
                    raise ValueError(
                        f'Sum over columns of stochastic matrix was out of valid\
                         bounds for marker1 {marker1}, marker2 {marker2}, for row {i} with value {_sum}')
        return ret


    def human_readable(self, rows_marker, cols_marker, matrix):

        rows_marker_N = self.num_states_for_marker(rows_marker)
        cols_marker_N = self.num_states_for_marker(cols_marker)

        row_labels = self.decode(rows_marker, list(range(rows_marker_N)))
        cols_labels = self.decode(cols_marker, list(range(cols_marker_N)))

        return pd.DataFrame(data=matrix, columns=cols_labels, index=row_labels)
