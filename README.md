# A Generic Framework for Hidden Markov Models on Biomedical Data

<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>


This repository holds the code for a prediction tool that is able to construct a HMM-based model solely from multivariate data and a descriptive configuration file. The model construction is completely automated, for an example, please take a look at the included tutorial.ipynb file.

# Overview

## What is the model able to predict?

The model does not predict sequences of truly hidden states as a HMM would, but is able to predict sequences of observable states from data. Thus, the model
is able to provide any prediction capability that a HMM would offer as well, the difference being that this model predicts observable states.

For further details, please read Section 5.2 Model Capabilities and Usecases in the provided thesis.

# Installation and Usage

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

# Contact, Supporters, Contributions and Contributing

The development of this work was carried out by several organizations:

| Organization                                             | Grant           |
|----------------------------------------------------------|---------------------------------|
| [University of Koblenz](https://www.uni-koblenz-landau.de/de/koblenz/fb3/organisation/personen/mathe/rockenfeller/robert-rockenfeller-fb3)        | Postdoc fellowship of the German Academic Exchange Service (DAAD)   |
| University of the Sunshine Coast, Queensland, Australia | |
| University of Queensland, Brisbane, Australia |  | 

If you have questions, please use the GitHub discussions feature at
https://github.com/rfechner/generic-hmm/issues/new.




# Citation

If you use this work, please cite:

```bibtex
```
