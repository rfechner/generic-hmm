from preprocessing import Preprocessor

path_to_config = "../../data/train.ini"
path_to_data = "../../data/train.csv"

prep = Preprocessor()
prep.process(path_to_config=path_to_config, path_to_data=path_to_data, csv_delimiter=',')


from pvault import ProbabilityVault

pv = ProbabilityVault(prep)
pv.extract_probabilities()

import pandas as pd
from model import DynamicNaiveBayesClassifier

dnbc = DynamicNaiveBayesClassifier(pv)

path_to_observation = "../../data/obs.csv"

obs_df = pd.read_csv(path_to_observation, delimiter=',')

post_distr = dnbc.posterior_distribution(hidden_marker='diagnosis', layers=['motoric', 'visit_specific'], observation=obs_df)

N = post_distr.shape[1]
M = post_distr.shape[0]
series = pd.Series(range(M))

series = prep.decode_series(marker='diagnosis', series=series)

import matplotlib.pyplot as plt

plt.stackplot(range(N), post_distr, labels=series)
plt.legend()
plt.show()