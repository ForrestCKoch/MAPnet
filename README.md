# COMP9417 Assignment Topic 0
## Forrest Koch (z3463797)

The source code and results for this project can be obtained from the [MAPnet](https://github.com/ForrestCKoch/MAPnet) github 
repository.  Data is not available for download, however, a live-demo can be provided upon request.

* `development_log.md`: This file served as a diary to occasionally log progress throughout the development process.  It begins
with a brief description of the "Framework" as well as the expected folder structure of datasets.  These topics may be of 
value to the reader for understanding the project.

* `mapnet/`: This folder contains the source code files for the bulk of the project
    * `train.py`: This is the main run script implementing the training loop and evaluation proceedures. Run with `python3 train.py -h` to see the options.
    * `data.py`: This module implements the Dataset class and other helper functions used for data manipulation.
    * `model.py`: This module implements the `torch.nn.Module` defining the network architecture to be trained.
    * `defaults.py`: This file contains the default settings for a few global variables.
* `models/`: This folder contains information about the various models trained during the experiments.  After the filter tests (section 3.5), it was no longer possible to upload the `*.dat` files to the github repo.  Any missing data models can be supplied upon request, however, it measures around 88G in size and could not practically be uploade.
    * Data for each model is contained within it's own timestamped folder (YYYY-MM-DD_HH-MM-SS).  Inside each folder will be
        * Test
    * `lr_trials/`
