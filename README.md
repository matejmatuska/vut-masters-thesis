# GNNs for Malware Traffic Identification
This repository contains the code used for my master's thesis, which focuses on using Graph Neural Networks (GNNs) for malware traffic identification:

> Matuška, Matej. **Identification of Malicious Network Communication using Graph Neural Networks**. Online, master's thesis. Kamil Jeřábek (supervisor). Brno: Brno University of Technology, Faculty of Information Technology, 2025. Available at: <https://www.vut.cz/en/students/final-thesis/detail/163740>.

## Usage
### Dependency installation
#### ipfixprobe
Currently only a containerized version of ipfixprobe is supported, there are two containers, one layered on top of the other, to build the containers run:
```bash
make build-ipfixprobe
```

#### Python
First, make sure to install the required dependencies. You can do this by running:
```bash
pip install -r requirements.txt
pip install -r requirements-torch.txt 
```
or
```bash
make install-deps
```
This will install all the required packages for the project. However not all scripts require all dependencies, if you only want to process the dataset, only `requirements.txt` is required, although it still installs some unnecessary packages. If you want to run the training scripts, you will need to install `requirements-torch.txt` as well.


### Dataset preparation
Transformation from PCAPs to CSVs, filtering, cleaning and transformation for training is done using the scripts in the `scripts` directory. To run all these scripts in order, you can use the `runall.sh` script. This script will run the scripts in order. Make sure to follow the instructions in the `runall.sh` script to set the paths for the ipfixprobe and logger executables and path to the dataset and working directory correctly.
One the variables are set, you can run the script as follows:
```bash
./runall.sh /path/to/dataset /path/to/workdir
```
WARNING: the script will create the necessary directories in the working directory, make sure to use it in a clean directory to prevent pollution or accidental overwriting of existing data.

See the `runall.sh` script for more details on how to run the script and how the output is structured.

#### Individual data processing scripts
These are the scripts that are run by the `runall.sh` script. You can run them individually if you want to customize the process or run only a part of it.
All of the scripts print the usage when run without arguments.

- scripts/process_dataset.sh - is used within the ipfixprobe container to convert PCAPs to CSVs. Outputs:
  - the output directory contains the CSV files each in a separate per-family directory.

- scripts/clean_dataset.py - is used to clean the dataset, performing common IPs filtering and DNS and rDNS filtering. Outputs:
  - removed_dns.csv - contains a list of domain names flows containing which were removed from the dataset
  - removed_rdns.csv - contains rDNS flows that were removed from the dataset
  - dataset-clean.csv - contains the filtered dataset, before removing extreme packet count samples, DNS only flows and sttitching DNS uniflows to biflows.
  - dataset-stitched.csv - contains the final clean dataset.
  - dataset-stitched.parquet - contains the final clean dataset as a parquet file.

- scripts/prep_dataset.py - is used to prepare the dataset for training - selecting samples, extracting and normalizing features and splitting the dataset into training, validation and test sets.
  Parameters can be tuned inside the script. Outputs:
    - dataset-train.parquet - contains the training set as a parquet file, used in training
    - dataset-val.parquet - contains the validation set as a parquet file, used in training
    - dataset-test.parquet - contains the test set as a parquet file, used in training
    - dataset-train.csv - contains the training set, for humanreadability
    - dataset-val.csv - contains the validation set, for human readability
    - dataset-test.csv - contains the test set, for human readability

- scripts/stitch_dns.py - used by the clean_dataset.py script to stitch DNS uniflows to biflows. Should not be run directly.

#### Input data
This data is required by the scripts/clean_dataset.py script.
- data/filter_common_ips.csv - contains the list of common IPs to filter out from the dataset. This is used to remove common IPs that are not relevant for the analysis. To generate a fresh list uncomment the generation in the `clean_dataset.py` script. A fresh list requires manual modification.
- data/top-1m.csv - Cisco Umbrella's list of the top 1 million domains. This is used to filter out common domains that are not relevant for the analysis. Available at: <https://umbrella-static.s3-us-west-1.amazonaws.com/index.html>.
- data/filter_common_ips.csv - if the generation is enabled in clean_dataset.py, contains the list of common IPs to filter out from the dataset. This is used to remove common IPs that are not relevant for the analysis.

#### Other data
- data/Dockerfile.bin - Dockerfile with ipfixprobe installation
- data/Dockerfile.data - Dockerfile building on the ipfixprobe container. Adds the scripts/process_dataset.sh script and runs in when the container is started.

### Training
There are two ways to train the models:

1. Using the `train.py` script, which runs a single training and evaluation run. For example, to train the model with the default parameters, you can run:
   ```bash
   python train.py /path/to/dataset /path/to/model
   ```
   This will train the model on the dataset located at `/path/to/dataset` and the specified model.

   Hyperparameters can be specified using command line arguments. See `--help` for more details.

2. Using the `tune.py` script, which runs a hyperparameter tuning process using Optuna. This script will run multiple training and evaluation runs with different hyperparameters and save the best model.
   ```bash
   python tune.py /path/to/dataset /path/to/model
   ```
   This will run the hyperparameter tuning process on the dataset located at `/path/to/dataset` and the specified model.
   Parameters to customize the tuning process can be specified using command line arguments. See `--help` for more details.
   The script creates `models` directory in the working directory, where it stores the models for each trial.

NOTE: Both scripts required mlflow server to run. If you want to run the server locally, you can do so by running:
```bash
mlflow server
```
## Training-related source files
- models/ - contains the model source files
    - models/baseline.py - the baseline model
    - models/repr1.py - the graph representation 1 model
    - models/repr2.py - the graph representation 2 model
- dataset/__init__.py - contains the dataset class and the transformation to the graph representation
- train.py - the training script
- tune.py - the hyperparameter tuning script
- train_common.py - common training functions
- utils/__init__.py - utility functions
