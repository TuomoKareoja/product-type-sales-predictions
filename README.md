# product-type-sales-prediction

Ubiqum Data Science Bootcamp project. Predicting future sales of different product types based on historical data.

## Project Organization

```
├── data
│   ├── clean          <- Data that has been cleaned of any clear errors.
│   ├── predictions    <- predictions made with models.
│   ├── processed      <- The final data sets for modeling.
│   └── raw            <- The original data.
│
├── notebooks          <- Jupyter notebooks and other notebooklike objects.
│    ├── exploratory   <- Lab books and work in progress. NOT IN remote repository.
│    └── reports       <- Notebooks meant for others to read. IN remote repository.
│
├── reports
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── main           <- LaTeX, PDF, etc.
│
├── src                <- Source code for use in this project.
│   ├── data           <- Scripts to download and generate data.
│   ├── models         <- Scripts to train models and then use models to make
│   │                     predictions. Hyperparameter tuning done in notebooks.
│   └── visualization  <- Scripts to create exploratory visualizations
│
├── .gitignore
├── environment.yml    <- Conda environment file for reproducing the environment.
├── LICENSE.md         <- MIT Licence
├── Makefile           <- Makefile with commands to recreate the analysis.
└── README.md
```

### Prerequisites

Conda

### Recreating the Analysis

1. Recreating the environment

```bash
make create_environment
```

2. Activate the conda environment (the rest of the make commands don't work without having the enviroment manually activated)

```bash
conda activate product-type-sales-prediction
```

3. Cleaning the data

```bash
make clean_data
```

4. "Building features" (really for this project just dropping one row and formatting the data for easy use)

```bash
make build_features
```

5. Training and predicting with the final models

```bash
make predict
```

6. Visualizing the predictions

```bash
make visual
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
