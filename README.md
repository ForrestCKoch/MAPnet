# COMP9417 Assignment Topic 0
## Forrest Koch (z3463797)

The source code and results for this project can be obtained from the [MAPnet](https://github.com/ForrestCKoch/MAPnet) github 
repository.  Data is not available for download, however, a live-demo can be provided upon request.

* `development_log.md`: This file served as a diary to occasionally log progress throughout the development process.  It begins
with a brief description of the "Framework" as well as the expected folder structure of datasets.  These topics may be of 
value to the reader for understanding the project.
* `README.md`: this README file
* `report.Rmd`: R-markdown file used to write the report
* `report.pdf`: the pdf version of the report
* `references.bib`: BibTex reference file
* `requirements.txt`: the module requirements for this project (can install with pip)
* `mapnet/`: This folder contains the source code files for the bulk of the project
    * `train.py`: This is the main run script implementing the training loop and evaluation proceedures. Run with `python3 train.py -h` to see the options.
    * `data.py`: This module implements the Dataset class and other helper functions used for data manipulation.
    * `model.py`: This module implements the `torch.nn.Module` defining the network architecture to be trained.
    * `defaults.py`: This file contains the default settings for a few global variables.
* `models/`: This folder contains information about the various models trained during the experiments.  After the filter tests (section 3.5), it was no longer possible to upload the `*.dat` files to the github repo.  Any missing data models can be supplied upon request, however, it measures around 88G in size and could not practically be uploade.
    * Data for each model is contained within it's own timestamped folder (YYYY-MM-DD_HH-MM-SS).  Inside each folder will be:
        * `arguments.txt`: This file details the arguments supplied to the program call
        * `loss.csv`: This file contains information about the training performance.  It is in the form [epoch],[lr],[train err],[test_err]
        * `*.dat`: These files are saved `MAPnet` modules.  They can be loaded with the `torch.load` function.
    * `lr_trials/`: contains the models trained for section 3.2
    * `layer_trials/`: contains the models trained for section 3.3
    * `decay_test/`: contains the models train for section 3.4
    * `filters_test/`: contains the models trained for section 3.5
    * `optims/`: contains the models trained for section 3.6
    * `final_model/`: contains the final model used for section 3.7
* `results/`: contains the spreadsheets used for analysis in the report
    * `csvs/lr_comparison`: used for section 3.2
    * `csvs/layer_comparison`: used for section 3.3
    * `decay_results.csv`: used for section 3.4
    * `filters_test.csv`: used for section 3.5
    * `weight_decay_results.csv`: used for section 3.6
    * `validation_ages.csv`: used for section 3.7
