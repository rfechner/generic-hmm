import configparser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from source.pyth.Preprocessing.preprocessing import Preprocessor
from source.pyth.Preprocessing.pvault import ProbabilityVault
from source.pyth.Model.model import RHMM
from source.pyth.Model.Interface.abstract_controller import AbstractController

class Controller(AbstractController):

    def __init__(self, debug = False):
        super().__init__()

        self.initialized = False
        self.debug = debug

    def __approx_stationary_distr(self, path_to_observation : str,
                                  csv_delimiter : str,
                                  layers : [str],
                                  hidden_marker : str,
                                  timesteps : int):

        if not self.initialized:
            print("Controller was not initialized yet.")
            return

        if not os.path.isfile(path_to_observation) \
                or not path_to_observation.endswith(".csv"):
            print("Please supply a valid .csv file and a correct path.")
            return

        self.__check_valid_layer_marker_combination(layers, hidden_marker)

        post_distr = self.__calc_post_distr(path_to_observation=path_to_observation,
                                            csv_delimiter=csv_delimiter,
                                            layers=layers,
                                            hidden_marker=hidden_marker)


        current_distribution = post_distr[:, -1]
        anterior_distr = self.model.iter_approx_stationary_distr(hidden_marker=hidden_marker,
                                                                 current_distr=current_distribution,
                                                                 timesteps=timesteps)



        return anterior_distr

    def __optimal_state_sequence(self, path_to_observation : str, csv_delimiter : str, layers : [str], hidden_marker : str):

        if not self.initialized:
            print("Controller was not initialized yet.")
            return

        if not os.path.isfile(path_to_observation) \
                or not path_to_observation.endswith(".csv"):
            print("Please supply a valid .csv file and a correct path.")
            return

        self.__check_valid_layer_marker_combination(layers, hidden_marker)

        observation = pd.read_csv(path_to_observation, delimiter=csv_delimiter)
        optimal_sequence = self.model.predict(hidden_marker=hidden_marker,
                                    layers=layers,
                                    observation=observation)

        return optimal_sequence

    def __calc_post_distr(self, path_to_observation: str, csv_delimiter: str, layers: [str],
                                    hidden_marker: str):
        if not self.initialized:
            print("Controller was not initialized yet.")
            return

        if not os.path.isfile(path_to_observation) \
                or not path_to_observation.endswith(".csv"):
            print("Please supply a valid .csv file and a correct path.")
            return

        self.__check_valid_layer_marker_combination(layers, hidden_marker)

        observation = pd.read_csv(path_to_observation, delimiter=csv_delimiter)
        distr = self.model.posterior_distribution(hidden_marker=hidden_marker,
                                                  layers=layers,
                                                  observation=observation)
        return distr

    def plot_anterior_distribution(self, path_to_observation : str,
                                    csv_delimiter : str,
                                    layers : [str],
                                    hidden_marker : str,
                                    timesteps : int):


        anterior_distr = self.__approx_stationary_distr(path_to_observation=path_to_observation,
                                                        csv_delimiter=csv_delimiter,
                                                        layers=layers,
                                                        hidden_marker=hidden_marker,
                                                        timesteps = timesteps)

        num_hidden_states = self.pv.num_states_for_marker(hidden_marker)
        timesteps = anterior_distr.shape[1]

        hidden_state_labels = self.pv.decode(marker=hidden_marker, series=list(range(num_hidden_states)))
        plt.stackplot(range(timesteps), anterior_distr, labels=hidden_state_labels)
        plt.xlabel("Timesteps")
        plt.ylabel(f"Distribution over marker {hidden_marker}")
        plt.legend(loc=(1.04, 0))
        plt.show()

        return anterior_distr


    def optimal_state_sequence(self, path_to_observation : str,
                                    csv_delimiter : str,
                                    layers : [str],
                                    hidden_marker : str):

        opt_seq = self.__optimal_state_sequence(path_to_observation=path_to_observation,
                                                csv_delimiter=csv_delimiter,
                                                layers=layers,
                                                hidden_marker=hidden_marker)

        decoded_sequence = self.pv.decode(hidden_marker, opt_seq)

        df = pd.read_csv(path_to_observation, delimiter=csv_delimiter)
        df[f"Predicted state for {hidden_marker}"] = decoded_sequence
        return df

    def plot_posterior_distribution(self, path_to_observation : str,
                                    csv_delimiter : str,
                                    layers : [str],
                                    hidden_marker : str):

        distr = self.__calc_post_distr(path_to_observation=path_to_observation,
                                       csv_delimiter=csv_delimiter,
                                       layers=layers,
                                       hidden_marker=hidden_marker)

        num_hidden_states = self.pv.num_states_for_marker(hidden_marker)
        timesteps = distr.shape[1]

        hidden_state_labels = self.pv.decode(marker=hidden_marker, series=list(range(num_hidden_states)))
        plt.stackplot(range(timesteps), distr, labels=hidden_state_labels)
        plt.xlabel("Timesteps")
        plt.ylabel(f"Distribution over marker {hidden_marker}")
        plt.legend(loc=(1.04, 0))
        plt.show()

        return distr

    def construct(self, path_to_data : str, path_to_config : str, csv_delimiter = ","):
        self.prep = Preprocessor(debug=self.debug)
        self.prep.process(path_to_config=path_to_config,
                     path_to_data=path_to_data,
                     csv_delimiter=csv_delimiter)

        self.pv = ProbabilityVault(self.prep, debug=self.debug)
        self.pv.extract_probabilities()

        self.model = RHMM(self.pv, debug=self.debug)

        self.initialized = True

        return self.model

    def validate(self, path_to_validation_data : str, csv_delimiter : str, layers : [str], hidden_marker : str):
        if not self.initialized:
            print("Controller was not initialized yet.")
            return

        if not os.path.isfile(path_to_validation_data) \
                or not path_to_validation_data.endswith(".csv"):
            print("Please supply a valid .csv file and a correct path.")
            return

        self.__check_valid_layer_marker_combination(layers, hidden_marker)

        validation_df = pd.read_csv(path_to_validation_data, delimiter=csv_delimiter)
        validation_samples = self.prep.group_df(validation_df)

        f1_score = self.model.validate(groups=validation_samples, layers=layers, hidden_marker=hidden_marker)

        print(f"The multiclass F1-Score was calculated. The resulting score is an average over all samples. "
              f"F1-Score : {f1_score}")

        return f1_score

    def kfold_cross_validation(self, k : int,
                            path_to_data : str,
                            path_to_config : str,
                            csv_delimiter : str,
                            layers :[str],
                            hidden_marker : str):

        temp_dir_name = os.path.join(os.curdir, "TEMP_FOLDER")

        if not os.path.isdir(temp_dir_name):
            os.mkdir(temp_dir_name)
        else:
            raise RuntimeError(f"temporal folder {temp_dir_name} already exists. Please delete the folder and try again.")

        cparser = configparser.ConfigParser()
        cparser.read(path_to_config)

        try:
            groupby = cparser["markerconfig_metainfo"]["groupby"]
        except:
            raise KeyError("The groupby key wasn't found inside your .ini file. k-fold cross validation could not continue.")

        def __del_temp_files(temp_dir_name : str):
            for filename in os.listdir(temp_dir_name):
                if filename.endswith('.csv'):
                    os.remove(os.path.join(temp_dir_name, filename))
            return

        def __cleanup(temp_dir_name : str):

            __del_temp_files(temp_dir_name)
            os.rmdir(temp_dir_name)
            return

        def __train_validate_split(chunk_length : int, groups : [pd.DataFrame], directory : str):

            indeces = np.random.permutation(len(groups))
            train_idx,validation_idx = indeces[chunk_length:],indeces[:chunk_length]

            validate_df = pd.concat(list(map(groups.__getitem__, validation_idx)))
            train_df = pd.concat(list(map(groups.__getitem__, train_idx)))

            validate_path = os.path.join(directory, "validate.csv")
            validate_df.to_csv(path_or_buf=validate_path, index=False)

            train_path = os.path.join(directory, "train.csv")
            train_df.to_csv(path_or_buf=train_path, index=False)

            return train_path, validate_path

        scores = []
        temp_df = pd.read_csv(path_to_data, delimiter=csv_delimiter).groupby(groupby)
        groups = [temp_df.get_group(x) for x in temp_df.groups]
        chunk_length = len(groups) // k

        for i in range(k):
            train_path,validate_path = __train_validate_split(chunk_length=chunk_length,
                                                              groups=groups,
                                                              directory=temp_dir_name)
            try:

                self.construct(path_to_data=train_path,
                           path_to_config=path_to_config,
                           csv_delimiter=csv_delimiter)

            except:
                self.initialized = False
                __cleanup(temp_dir_name)
                raise

            f1_score = self.validate(path_to_validation_data=validate_path,
                          csv_delimiter=csv_delimiter,
                          layers = layers,
                          hidden_marker=hidden_marker)

            scores.append(f1_score)
            __del_temp_files(temp_dir_name)

        __cleanup(temp_dir_name)

        print(f"The mean score is {np.mean(scores)} with a standart deviation of {np.std(scores)}.")

        return scores



    def __check_valid_layer_marker_combination(self, layers : [str], hidden_marker : str):
        layerinfo = self.pv.get_layerinfo_from_layers(hidden_marker=hidden_marker, layers=layers)

        if len(layerinfo.keys()) == 0 or \
            sum([len(layerinfo[layername]['markers'].keys()) for layername in layerinfo.keys()]) == 0:
            raise RuntimeError("Given layers cannot provide enough different markers in order to predict the hidden state."
                               " Please increase the number of layers used for prediction.")


















