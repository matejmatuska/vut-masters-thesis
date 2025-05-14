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
