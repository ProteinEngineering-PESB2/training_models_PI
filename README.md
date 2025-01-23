# Demo library for train predictive models in protein engineering and bioinformatic tasks

This repository facilitates the training of predictive models for protein engineering and bioinformatic tasks.

Two type of models are considered, including classification and regression tasks. The repository facilitates the evaluation of different supervised learning algorithm and the exploration of different hyperparameters. Also, the library facilitates:

- Evaluate a trained model
- Export trained models
- Load and use trained models
- Tuning hyperparameters with different strategies
- Characterize the trained models through different plots and visualization

## Before to start

The folder [dataset_example](dataset_example) contains a dataset of antiviral peptides with half life measures collected during the thesis of one of our members. This half life values also are associated in different systems and were classified on three categories concerning the value of the half life.

The jupyter notebook [demo_preparation_data.ipynb](notebooks/demo_preparation_data.ipynb) facilitates the preprocessing and generation of the following datasets:

- A dataset for classification task. To develop classification models to predict the category of half life in AVP peptides
- Three datasets for regression tasks, generating datasets per each type of category.

Each generated dataset is processed by applying numerical representation strategies. In this case, it was used the library [protein_representation_strategies](https://github.com/ProteinEngineering-PESB2/protein_representation_strategies) implemented by us.

The following command line was executed to obtain the processed embedding:

```
python runing_esm.py -d input_csv -c column_with_seq -o name_outout -r list of columns to ignore -e 1
```

Please, see more details in the library [protein_representation_strategies](https://github.com/ProteinEngineering-PESB2/protein_representation_strategies) in particular about how to use and the different available strategies.



