from controller import Controller
import os
import warnings
from tqdm import tqdm
import numpy as np


warnings.filterwarnings('ignore')

path_to_data = os.path.join('./data/@jenny/data/merged_sumonly.csv')
path_to_config = os.path.join('./data/@jenny/config.ini')
print(os.path.isfile(path_to_data), os.path.isfile(path_to_config))

c = Controller()

# without hyperparam optim
prev_scores = c.kfold_cross_validation(k=10,
                         path_to_data=path_to_data,
                         path_to_config=path_to_config,
                         csv_delimiter=',',
                         hidden_marker='dissta',
                         layers=['MEDICAL'])

other_scores_mean, other_scores_std = [], []
for i in tqdm(range(100)):
        
    # hyperparam optim
    new_layerweights, optim_results  = c.optim_layerweights(path_to_data=path_to_data,
                            path_to_config=path_to_config,
                            csv_delimiter=',',
                            hidden_marker='dissta',
                            layers=['MEDICAL'])

    other_scores = c.kfold_cross_validation(k=10,
                            path_to_data=path_to_data,
                            path_to_config=path_to_config,
                            csv_delimiter=',',
                            hidden_marker='dissta',
                            layers=['MEDICAL'],
                            new_layerweights=new_layerweights)
    other_scores_mean.append(np.mean(other_scores))
    other_scores_std.append(np.std(other_scores))


