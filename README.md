## Usage
The transformation from PCAPs to CSVs and the cleaning of the dataset is done in two steps to allow rerunning the cleaning step without having to reprocess the PCAPs:

### `process_dataset.sh`
Process PCAPs into CSVs using ipfixprobe

Usage: `./process_dataset.sh <pcap_dataset> <output_dir>`
- `<pcap_dataset>`: Path to the directory containing subdirectories for each family that contain the PCAPs
- `<output_dir>`: Path to the directory where the CSVs will be saved, each family will have a subdirectory containing the CSVs created automatically
- set the `IPFIXPROBE_PATH` and `LOGGER_PATH` variable in the script to the path of the ipfixprobe and logger executables
- can also change the `N` variable to the number of parallel processes to run

### `clean_dataset.py`
Load, clean and merge the dataset created by `process_dataset.sh` into a single CSV `dataset.csv`

Usage: `python clean_dataset.py <dataset_dir> <output_dir>`
- `<dataset_dir>`: Path to the directory containing the CSVs created by `process_dataset.sh`
- `<output_dir>`: Path to the directory where the cleaned cleaned dataset will be saved
