# Pipeline for the generic generation of a HMM-based prediction model for multivariate sequences

This repository holds the (public) files for my bachelors thesis 'Generic Generation of Hidden Markov Models and an Application to Medical Data'. 

## What is the purpose of this repository?

This repository holds the code for a prediction tool that is able to construct a HMM-based model solely from multivariate data
and a descriptive configuration file. The model construction is completely automated, for an example, please take a look at the included tutorial.ipynb file.

## What is the model able to predict?

The model does not predict sequences of truly hidden states as a HMM would, but is able to predict sequences of observable states from data. Thus, the model
is able to provide any prediction capability that a HMM would offer as well, the difference being that this model predicts observable states.

For further details, please read Section 5.2 Model Capabilities and Usecases in the provided thesis.

## How does one use the prediction pipeline?

1) Clone/Download this repository.
2) Navigate to the folder on your machine using the commandline tool
3) (optional) create a new python environment
4) (python 3.8X required) Enter the following command into your commandline

```bash
pip install -r requirements.txt
```

5) start the tutorial notebook by entering the following command

```bash
jupyter-notebook
```

