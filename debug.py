from controller import Controller
import os
import warnings


#warnings.filterwarnings('ignore')

path_to_data = os.path.join('./data/@jenny/data/merged_sumonly.csv')
path_to_config = os.path.join('./data/@jenny/config.ini')
print(os.path.isfile(path_to_data), os.path.isfile(path_to_config))

c = Controller()

# 10-fold cross validation
f1_scores = c.optim_layerweights(path_to_data=path_to_data,
                         path_to_config=path_to_config,
                         csv_delimiter=',',
                         hidden_marker='dissta',
                         layers=['MEDICAL'])