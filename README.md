# product-type-sales-prediction

Ubiqum Data Science Bootcamp project. Predicting future sales of different product types based on historical data.

## Project Organization

```
├── data
│   ├── clean          <- Data that has been cleaned of any clear errors.
│   ├── interim        <- Intermediate step for big data transformations.
│   ├── predictions    <- predictions made with models.
│   ├── processed      <- The final data sets for modeling.
│   └── raw            <- The original data.
│
├── docs               <- A default Sphinx project. See sphinx-doc.org.
│
├── models
│    ├── final         <- Models trained with all data.
│    └── train         <- Models trained with training data.
│
├── notebooks          <- Jupyter notebooks and other notebooklike objects.
│    │                    Names start with a number for ordering
│    │                    and have a `_` delimited description,
│    │                    e.g. `1.0_initial_data_exploration`.
│    ├── exploratory   <- Lab books and work in progress. NOT IN remote repository.
│    └── reports       <- Notebooks meant for others to read. IN remote repository.
│
├── reports
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── main           <- LaTeX, PDF, etc.
│
├── src                <- Source code for use in this project.
│   │
│   ├── data           <- Scripts to download and generate data.
│   │   └── make_datasets.py
│   │
│   ├── models         <- Scripts to train models and then use models to make
│   │   │                 predictions. Hyperparameter tuning done in notebooks.
│   │   ├── predict_comparisons.py
│   │   ├── predict_final.py
│   │   ├── train_models_comparison.py
│   │   └── train_models_final.py
│   │
│   └── visualization  <- Scripts to create exploratory visualizations
│       └── visualize.py
│
├── .env               <- variables for dotenv. NOT in version control
├── .gitignore
├── environment.yml    <- Conda environment file for reproducing the environment.
├── LICENSE.md         <- MIT Licence
├── Makefile           <- Makefile with commands to recreate the analysis.
└── README.md
```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```bash
Give examples
```

### Recreating the Analysis

What commands need to be run the critical parts of the analysis

```bash
Give examples
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
