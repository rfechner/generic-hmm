import csv
import os

import numpy as np
import pandas as pd

"""
markers:
patientid
date

mobility:
g, mg, m, mb, b, s (good, med-good, med, med-bad, bad, severe)
motoric:
g, mg, m, mb, b, s (good, med-good, med, med-bad, bad, severe)
neuro:
g, mg, m, mb, b, s (good, med-good, med, med-bad, bad, severe)
diagnosis:
g, mg, m, mb, b, s (good, med-good, med, med-bad, bad, severe)
"""

diagnosis_init_state = np.array([0.6, 0.25, 0.1, 0.03, 0.02, 0.0 ])

diagnosis_motoric_obs = np.matrix("0.7 0.2 0.1 0 0 0;\
                                    0.0 0.5 0.3 0.2 0 0;\
                                    0.0 0.0 0.6 0.3 0.1 0;\
                                    0.0 0.0 0.0 0.7 0.2 0.1;\
                                    0 0.0 0.0 0. 0.8 0.2;\
                                    0 0 0.0 0.0 0.0 1")

diagnosis_mobility_obs = np.matrix("0.6 0.3 0.1 0 0 0;\
                                    0.55 0.25 0.1 0.1 0 0;\
                                    0.3 0.3 0.1 0.2 0.1 0;\
                                    0 0.1 0.4 0.2 0.2 0.1;\
                                    0 0 0.2 0.3 0.3 0.2;\
                                    0 0 0.1 0.1 0.4 0.4")

diagnosis_neuro_obs = np.matrix("0.4 0.4 0.1 0.1 0. 0;\
                                    0.0 0.3 0.3 0.2 0.2 0;\
                                    0.0 0.0 0.5 0.3 0.2 0;\
                                    0.0 0.0 0.0 0.5 0.4 0.1;\
                                    0 0.0 0.0 0. 0.6 0.4;\
                                    0 0 0.0 0.0 0.0 1")

diagnosis_transitions = np.matrix("0.5 0.4 0.1 0 0 0;\
                                    0. 0.5 0.4 0.1 0 0;\
                                    0 0. 0.5 0.45 0.05 0;\
                                    0 0 0. 0.4 0.5 0.1;\
                                    0 0 0 0. 0.8 0.2;\
                                    0 0 0 0 0. 1")

names = [s.strip() for s  in "good, med-good, med, med-bad, bad, severe".split(",")]


def sample(distr: np.array):
    return np.random.choice(names, p=distr)


NUM_ITERATIONS = 1
path = os.path.join(os.curdir, 'thesis_simple_test1', 'single1.csv')
with open(path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    # write header row
    writer.writerow(['id', 'date', 'mobility', 'motoric', 'neuro', 'diagnosis'])

    for i in range(NUM_ITERATIONS):
        measurements = np.random.randint(15, 16)
        time = pd.Timestamp(f"{np.random.randint(2020, 2023)}-{np.random.randint(1, 12)}-{np.random.randint(1, 28)}")
        diagnosis_state = np.random.choice(names, p=diagnosis_init_state)

        for j in range(measurements):
            current_diag_index = names.index(diagnosis_state)

            mobility = sample(np.array(diagnosis_mobility_obs[current_diag_index, :])[0])
            motoric = sample(np.array(diagnosis_motoric_obs[current_diag_index, :])[0])
            neuro = sample(np.array(diagnosis_neuro_obs[current_diag_index, :])[0])

            writer.writerow([str(i), str(time.strftime('%Y-%m-%d')), mobility, motoric, neuro, diagnosis_state])

            diagnosis_state = sample(np.array(diagnosis_transitions[current_diag_index, :])[0])
            time += pd.Timedelta(days=np.random.choice([30, 5, 2, 1], p=[0.01, 0.09, 0.2, 0.7]),
                                 hours=np.random.choice([12, 4, 1], p=[0.1, 0.2, 0.7]))
