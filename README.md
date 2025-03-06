# Demo library for train predictive models in protein engineering and bioinformatic tasks

This repository facilitates the training of predictive models for protein engineering and bioinformatic tasks.

Specifically, the repository has implemented different modules and jupyter notebooks to preparing, encoding, training, and use predictive models.

## A traditional pipeline to develop predictive models using sequence-based approaches

The next figure demonstrates a brief simple pipeline to develop predictive models using machine learning and protein sequences as input.

![alt text](figures/data_driven_simple.png "Demo Pipeline")

The process starts with a data collection, then the data is processed and numerical representation strategies need to be applied to represent the sequences for training process. Then, a machine learning model is trained. Traditionally, the dataset is divided into training, validation, and testing. The training dataset is used to fit the hyperparameters of the model, validation and testing datasets are used to evaluate the performances of the model. Alternative methods included the use of cross validation to prevent overfitting in the case of low-N regime datasets.

## Demo datasets

This repository has two demonstrative datasets:

1. **Antimicrobial peptides**: This dataset facilitates the identification of peptides with the antimicrobial activity and peptides without the antimicrobial activity. This dataset is already prepared (see Folder [antimicrobial](raw_data/Antimicrobial/)). The preparation of the dataset implied the discrepance reduction, homology redundancy deletion, and split the data into training, validation, and testing dataset.

2. **Protein solubility**: This dataset contains information about the solubility of the protein sequences (in percentage). Moreover, this dataset has been preprocessed removing inconsistences and redundancy. In this case, the redundancy was removed by applying CD-Hit on range of the data by quantiles. Besides, the data was divided using the notebook [demo_split_data.ipynb](notebooks/demo_split_data.ipynb). 

## Source code and implementation strategies

## Examples

## Coming soon

- Fine tuning for regression models
- Alternative strategies to develop predictive models