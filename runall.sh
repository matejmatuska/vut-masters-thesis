#!/bin/bash
## runall.sh
# This script is used to run the entire process of the dataset preparation and
# subsequent training.

### PREPARATION
#### DATASET
# The dataset should be extracted to a directory with the following structure:
#   $1/extracted/2024072?/<family>/*.{pcapng,pcap,...}
# NOTE: there is no 'Malware/' part before the date directories,
#   you can use the following command to extract the dataset:
#   tar -xvf --strip-components=1 dataset.tar -C $1/extracted

#### IPFIXPROBE CONTAINER
# The ipfixprobe container must be available.
# The default image used is localhost/process-data:latest. You can change the
# image by setting the CONTAINER variable.

### DIRECTORY STRUCTURE
# The script will create the following files and directories:
#  - $1/combined - combined date directories with duplicate files removed
#  - $1/csvs - pcaps converted to csvs
#  - $1/out - the final dataset in the form of a single CSV and
#  otherinteresting outputs from the filtering process
#  - $1/train - outputs from the training process
# WARNING: the $1/extracted directory will be modified in place - duplicates
# will be removed

### USAGE
# ./runall.sh <work-dir>
#    work-dir - the directory containing the extracted subdirectoryt where the
#               dataset is extracted and the outputs will be written

set -o pipefail

DATASET_DIR="$1"
if [ -z "$DATASET_DIR" ]; then
    echo "Usage: $0 <dataset-dir> <work-dir>"
    exit 1
fi

WORKDIR="$2"
if [ -z "$WORKDIR" ]; then
    echo "Usage: $0 <dataset-dir> <work-dir>"
    exit 1
fi

CONTAINER=${CONTAINER:-localhost/process-data:latest}
LOG_FILE="$WORKDIR/log-$(date +"%Y-%m-%dT%H:%M:%S").txt"

EXTRACT_DIR="$WORKDIR/extracted"
COMBINE_DIR="$WORKDIR/combined"
PRE_FILTER_DIR="$WORKDIR/csvs"
OUT_DIR="$WORKDIR/out"

mkdir -p "$EXTRACT_DIR" "$COMBINE_DIR" "$PRE_FILTER_DIR" "$OUT_DIR" || exit 1


function stage {
    # print a stage message in green
    echo -e "\033[32mStage: $1\033[0m"
}

pushd "$DATASET_DIR" || exit 1

stage "Unpacking archives to $EXTRACT_DIR ..."
for i in {3..4}; do
    file="$DATASET_DIR/Malware-2024072$i.tar.gz"
    if [ ! -f "$file" ]; then
        echo "File $file not found, skipping."
        continue
    fi
    echo "Unpacking $file ..." | tee -a "$LOG_FILE"
    tar --strip-components=1 -xzvf "$file" -C "$EXTRACT_DIR" >> "$LOG_FILE" @>&1 || exit 1
done

for i in {5..8}; do
    file="$DATASET_DIR/Malware-2024072$i.tar"
    if [ ! -f "$file" ]; then
        echo "File $file not found, skipping."
        continue
    fi
    echo "Unpacking $file ..." | tee -a "$LOG_FILE"
    tar --strip-components=1 -xvf "$file" -C "$EXTRACT_DIR" >> "$LOG_FILE" 2>&1 || exit 1
done


stage "Removing duplicates in $EXTRACT_DIR ..."
find "$EXTRACT_DIR" -name '2407*' -type f -print0 |
    awk -F/ 'BEGIN { RS="\0" }
        { n=$NF; files[n] = files[n] $0 "\n"; count[n]++ }
        END {
            for (n in files) {
                split(files[n], arr, "\n");
                print "Keeping: " arr[1];
                for (i = 2; i <= length(arr); i++) {
                    if (arr[i] != "") {
                        print "Deleting duplicate: " arr[i];
                        system("rm -f \"" arr[i] "\"");
                    }
                }
            }
        }' >> "$LOG_FILE" 2>&1 || exit 1


stage "Combining date directories in $EXTRACT_DIR to $COMBINE_DIR ..."
# this will sometimes print out error because not all family directories are
# present in all date directories
for dir in "$EXTRACT_DIR"/2024072?; do
    for fam in "$dir"/*; do
        mkdir -p "$COMBINE_DIR/$(basename "$fam")"
        cp -r "$fam"/* "$COMBINE_DIR/$(basename "$fam")"
    done
done


stage "Converting files in $COMBINE_DIR to CSVs to $PRE_FILTER_DIR ..."
# TODO set the dirs better for the podman run
podman run --rm -ti -v "$WORKDIR":/data:z "$CONTAINER" \
    "$(basename "$COMBINE_DIR")" "$(basename "$PRE_FILTER_DIR")" >> "$LOG_FILE" 2>&1 || exit 1

popd || exit 1


stage "Filtering files in $PRE_FILTER_DIR to $OUT_DIR ..."
python scripts/clean_dataset.py "$PRE_FILTER_DIR" "$OUT_DIR" || exit 1


stage "Preparing for training $OUT_DIR/dataset-clean.csv -> $OUT_DIR ..."
python scripts/prep_dataset.py "$OUT_DIR/dataset-stitched.csv" "$OUT_DIR" || exit 1

stage "Done. Outputs written to $OUT_DIR"

# TODO add the training process
