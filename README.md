KUPPNet
==============================

CNN-RNN based phosphorylation site prediction tool

Basic usage for predictions
(execution from project dir is assumed)
```
python kuppnet.py predict data/test/test_predict/100seqsPSP.fasta -v 1
```
this will print out prediction to stdout.

for now tested only for with GPU
a lot more (including jupyter notebooks walkthroughs) is coming soon

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── test           <- Data for testing
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── hparams        <- Hyperparameters (separate for every model)
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── load_model.py
    │   │  
    │   │
    │   ├── metrics.py
    │   │  
    │   │
    │   ├── predict.py
    │   │
    │   │  
    │   │── prepare_input.py
    │   │
    │   │── structures.py
    │   │
    │   │── utils.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


