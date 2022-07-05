import warnings
import pandas as pd
import numpy as np
from numpy.linalg import norm
from itertools import chain
from pvault import ProbabilityVault

class DynamicNaiveBayesClassifier:
    
    def __init__(self, pv : ProbabilityVault):
        
        self._pv = pv
        
    def anterior_distribution(self, marker : str, distr : dict, timesteps : int):
        pass
    
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

        markers = list(chain(*[layerinfo[l]['markers'].keys() for l in layers]))
        
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
            
            distributions[observation_marker] = self.forward_probs(hidden_marker, observation_marker, observation[observation_marker])
            
        warnings.warn('EXPERIMENTAL CODE: layer weights are not applied, since the code for layer-weights is still missing. \
        The distr of weights over the layers is an equal distribution for now.')
        
        
        # 3) apply weights to the distributions, sum up the distributions
        layer_weight = 1.0 / len(layers)
        for layer in layerinfo.keys():
            for m, d in layerinfo[layer]['markers'].items():

                distributions[m] *= d['weight'] * layer_weight
                
        summed_distr = sum(distributions.values())
        
        # 4) return final distribution
        return summed_distr
                
        
        
    def forward_probs(self, hidden_marker, observation_marker, observation_sequence : pd.Series):
        
        # 1) encode series of values
        obs = self._pv.encode(observation_marker, observation_sequence)
        
        num_hidden_states = self._pv.num_states_for_marker(hidden_marker)
        timesteps = len(obs)
        
        # 2) create lookup table alpha for dynamic programming
        alpha = np.zeros((num_hidden_states, timesteps))
        
        
        for i in range(num_hidden_states):
            
            pi_i = self._pv.get_initial_prob(hidden_marker, i)
            b_i_O_0 = self._pv.get_observation_prob(hidden_marker, observation_marker, i, obs[i])
        
            alpha[i,0] = pi_i * b_i_O_0

        """
        We have to differentiate between three cases:
        
        1) the probabilities in the first column sum up to 1. everythings fine.
        2) the l1 norm of the first column is 0. -> We take a normal step in our markov chain. For the first
            col, this means assigning the initial state probabilities.
        3) the sum of probabilities lies somewhere in (0, 1) -> we have to normalize.
            This may occur, since we have a sparse observation/state transition space. Thus, often times
            the transition probability or the observation probability is 0.
        
        """
        l1norm = norm(alpha[:, 0], 1)
        eps = 1e-8

        if l1norm <= eps:
            for i in range(num_hidden_states):
                alpha[i, 0] = self._pv.get_initial_prob(marker=hidden_marker, state=i)

        elif 1 > l1norm and l1norm > eps:
            warnings.warn('Normalizing forward probabilities. This step should be discussed further.')
            alpha[:, 0] = alpha[:, 0] / l1norm


        for t in range(1, timesteps):
            for j in range(num_hidden_states):
                
                tmp_trans_probs = [self._pv.get_transition_prob(hidden_marker, i, j) for i in range(num_hidden_states)]
                _sum = sum([a*b for a,b in zip(alpha[:, t-1], tmp_trans_probs)])
                
                alpha[j, t] = _sum * self._pv.get_observation_prob(hidden_marker, observation_marker, j, obs[t])

            l1norm = norm(alpha[:, t], 1)

            if l1norm <= eps:
                warnings.warn(f"l1norm of column {t} was {l1norm}, this was an unexpected value. Something went wrong calculating prior\ "
                f"probabilities. The model will take a normal step in the markov chain.")

                for j in range(num_hidden_states):
                    tmp_trans_probs = [self._pv.get_transition_prob(hidden_marker, i, j) for i in range(num_hidden_states)]
                    alpha[j, t] = sum(np.array([a*b for a,b in zip(alpha[:, t-1], tmp_trans_probs)]))

                new_l1norm = norm(alpha[:, t], 1)

                if new_l1norm > 1.0 + eps or new_l1norm < 1 - eps:
                    raise ValueError(f"We encountered an unexpected error: After taking a normal step in the \ "
                    f"Markov Chain due to missing observation probabilities, we calculated a new l1norm: {new_l1norm}.\ "
                    f"This l1norm is out of the given bounds 1.0 > l1norm > 1.0 - {eps}.")

            elif 1 > l1norm and l1norm > eps:
                warnings.warn('Normalizing forward probabilities. This step should be discussed further.')
                alpha[:, t] = alpha[:, t] / l1norm

        return alpha
    
    def optimal_posterior_state_seq(self, marker : str, layers : [str], observation : pd.DataFrame):
        pass
    
    def optimize_weights(self):
        pass
    