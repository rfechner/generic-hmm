import warnings
import pandas as pd
import numpy as np

from tqdm import tqdm
from itertools import chain
from pvault import ProbabilityVault
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import f1_score


from base.abstract_model import AbstractModel


class RHMM(AbstractModel):

    def __init__(self, pv: ProbabilityVault, debug=False):
        
        super().__init__(pv)

        self._pv = pv
        self._debug = debug

    def iter_approx_stationary_distr(self, hidden_marker : str,
                                     current_distr : np.array,
                                     timesteps : int):

        transmat = np.transpose(self._pv.transition_matrix(hidden_marker))
        anterior_distr = np.zeros(shape=(transmat.shape[0], timesteps + 1))
        anterior_distr[:, 0] = current_distr

        for t in range(1, timesteps + 1):
            current_distr = transmat @ current_distr

            if self._debug:
                assert abs(sum(current_distr) - 1) <= 1e-10

            anterior_distr[:, t] = current_distr

        return anterior_distr




    def predict(self, hidden_marker : str, layers : [str], observation : pd.DataFrame):

        # 0) bring observation in temporal order
        dateid = self._pv.get_dateid_if_present()
        if dateid is not None:
            observation = observation.sort_values(by=dateid)

        # 1) get all markers from looping through sections of cparser
        layerinfo = self._pv.get_layerinfo_from_layers(hidden_marker, layers)

        markers = list(chain(*[layerinfo[l]['markers'].keys() for l in layerinfo.keys()]))

        unique_markers = set()

        for m in markers:
            if m not in unique_markers:
                unique_markers.add(m)
            else:
                warnings.warn(f'WARNING: It seems, that the marker {m} was found in more than one layer. Consecutive consultations \
                                            of markers are ignored. This may lead to unexpected behaviour in model predictions')

        # 2) calculate optimal state sequence according to each marker.
        optimal_sequences = {}

        for observation_marker in unique_markers:

            try:
                # try to encode the observation for the current marker
                encoded = self._pv.encode(marker=observation_marker, series=observation[observation_marker])
            except:
                # Encoding of observation has failed. We cannot infer an optimal sequence from an observation containing
                # an unknown state.
                warnings.warn(
                    f"We encountered an error while trying to encode the observation \n{observation[observation_marker]}"
                    f"This error occurs, when we encounter a new, before unseen state. Please either remove the observation with"
                    f"the unseen state, or adjust the prediction layers accordingly.")
                raise

            optimal_sequences[observation_marker] = \
                self.__single_marker_opt_seq(hidden_marker,
                                             observation_marker,
                                             encoded)

        distribution = np.zeros(shape=(self._pv.num_states_for_marker(hidden_marker), observation.shape[0]))

        # build weighted sum over optimal state sequences for different markers from different layers.
        for observation_marker, sequence in optimal_sequences.items():
            for i in range(sequence.shape[0]):

                layer = self.__get_layer_for_marker(layerinfo=layerinfo, marker=observation_marker)
                layer_weight = layerinfo[layer]['weight']
                marker_weight = layerinfo[layer]['markers'][observation_marker]['weight']
                distribution[sequence[i], i] += layer_weight * marker_weight

        optimal_sequence = [np.argmax(distribution[:, i]) for i in range(observation.shape[0])]

        return optimal_sequence

    def posterior_distribution(self, hidden_marker : str, layers : [str], observation : pd.DataFrame):
        """
        The posterior distribution over the states of a hidden_marker for the duration of the observation
        sequence is calculated with the forward-algorithm respectivly for each observation fragment.
        
        https://en.wikipedia.org/wiki/Forward_algorithm#Pseudocode
        
        The resulting distributions are weighted according to their respective weights specified inside
        the cparser of the ProbabilityVault object.
        
        A final distribution over the states for the duration of the observation is returned.
        
        hidden_marker: the identifier over the group of 'hidden states' we are calculating the distribution for
        layers: the groups of markers we are using as observation
        observation: total **NOT YET ENCODED** observation sequence containing all markers 
                        (possibly more than we need) with respective values for each timestep.
        
        
        0) bring observation in temporal order
        1) get all markers from looping through sections of cparser
        2) for each marker, call _post_distr_single to get the distribution over hidden states for one marker
        3) apply weights to the distributions, sum up the distributions
        4) return final distribution
        
        """
        
        # 0) bring observation in temporal order
        dateid = self._pv.get_dateid_if_present()
        if dateid is not None:
            observation = observation.sort_values(by=dateid)
        
        
        # 1) get all markers from looping through sections of cparser
        layerinfo = self._pv.get_layerinfo_from_layers(hidden_marker, layers)

        markers = list(chain(*[layerinfo[l]['markers'].keys() for l in layerinfo.keys()]))
        
        unique_markers = set()
        
        for m in markers:
            if m not in unique_markers:
                unique_markers.add(m)
            else:
                warnings.warn(f'WARNING: It seems, that the marker {m} was found in more than one layer. Consecutive consultations \
                        of markers are ignored. This may lead to unexpected behaviour in model predictions')
        
        # 2) for each marker, call forward_probs to get the distribution over hidden states for one marker
        distributions = {}
        
        for observation_marker in unique_markers:

            try:
                # try to encode the observation for the current marker
                encoded = self._pv.encode(marker= observation_marker, series=observation[observation_marker])
            except:
                # Encoding of observation has failed.
                raise ValueError(f"We encountered an error while trying to encode the observation \n {observation[observation_marker]} \n \ "
                                 f"This error occurs, when we encounter a new, before unseen state. Please either remove the observation with \ "
                                 f"the unseen state, or adjust the prediction layers accordingly.")

            # construct HMM from hmmlearn library. Call predict_proba, the result of the call will
            # be the distribution over hidden states for all timesteps
            distributions[observation_marker] = self.__single_marker_forward_probs(hidden_marker, observation_marker, encoded)


        warnings.warn('EXPERIMENTAL CODE: layer weights are not applied, since the code for layer-weights is still missing. \
        The distr of weights over the layers is an equal distribution for now.')
        
        
        # 3) apply weights to the distributions, sum up the distributions
        layer_weight = 1.0 / len(layerinfo.keys())
        for layer in layerinfo.keys():
            for m, d in layerinfo[layer]['markers'].items():

                distributions[m] *= d['weight'] * layer_weight
                
        summed_distr = sum(distributions.values())
        
        # 4) return final distribution
        return np.transpose(summed_distr)

    def __construct_hmm(self, hidden_marker : str,  obs_marker : str):

        num_hidden_states = self._pv.num_states_for_marker(hidden_marker)
        tmphmm = CategoricalHMM(n_components=num_hidden_states,
                                init_params="",
                                params="ste")

        # get initial state vector, transition matrix and emission matrix from pvault
        tmphmm.startprob_ = self._pv.initial_state_distribution(hidden_marker)
        tmphmm.transmat_ = self._pv.transition_matrix(hidden_marker)
        tmphmm.emissionprob_ = self._pv.emission_matrix(hidden_marker, obs_marker)

        return tmphmm

    def __reshape_obs(self, encoded_observation : np.ndarray):

        # bring observation in the right shape to process
        X = encoded_observation.reshape(-1, 1)
        if self._debug:
            assert len(X.shape) == 2 and X.shape[1] == 1

        return X

    def __single_marker_opt_seq(self, hidden_marker : str, obs_marker : str, encoded_observation : np.ndarray):

        tmphmm = self.__construct_hmm(hidden_marker, obs_marker)
        X = self.__reshape_obs(encoded_observation)
        optimal_sequence = tmphmm.predict(X)

        return optimal_sequence


    def __single_marker_forward_probs(self, hidden_marker : str, obs_marker : str, encoded_observation : np.ndarray):

        tmphmm = self.__construct_hmm(hidden_marker, obs_marker)
        X = self.__reshape_obs(encoded_observation)
        forward_probs = tmphmm.predict_proba(X)

        return forward_probs

    def validate(self, groups : [pd.DataFrame], layers : [str], hidden_marker : str):
        """
        groups: A list of single, unencoded DataFrames which are evaluated one by one.

        returns : A single f1-Score for all validation datasets.
        see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        In this particular case, we use the multiclass f1-Score since our classification has multiple classes (hidden states).
        The y_true labels are the actual sequences of the hidden_state marker of the samples, while
        the y_pred labels are the result of the predict() method call -> "optimal" state-sequence calculated
        via viterbi algorithm.
        """

        y_true = []
        y_pred = []

        for observation in tqdm(groups):
            try:
                optimal_sequence = self.predict(hidden_marker=hidden_marker, layers=layers, observation=observation)

            # if the method predict raised an exception, its likely because the observation contained a previously unseen state.
            except ValueError:
                warnings.warn('We encountered a problem with the prediction. The sample causing the error will be skipped.')
                continue


            try:
                actual_sequence = self._pv.encode(marker=hidden_marker, series=observation[hidden_marker])
            except:
                print(f"Encoding of observation sequence {observation[hidden_marker]} of marker {hidden_marker} encountered \ "
                      f"an error. Most likely the sequence contains a previously unseen state. The sample will be skipped.")
                continue

            if self._debug:
                assert len(optimal_sequence) == len(actual_sequence)

            y_pred = np.concatenate([y_pred, optimal_sequence])
            y_true = np.concatenate([y_true, actual_sequence])

        return f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    def __get_layer_for_marker(self, layerinfo : dict, marker):
        for layer, d in layerinfo.items():
            if marker in d['markers'].keys():
                return layer
        return None
