#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Expected 2 arguments!" >&2
    exit 1
fi

DATASET_PATH="$1"
OUTPUT_DIR="$2"

if [ ! -d "$DATASET_PATH" ]; then
    echo "Dataset path does not exist!" >&2
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory does not exist!" >&2
fi

IPFIXPROBE_CMD=/usr/local/bin/ipfixprobe
LOGGER_CMD=/usr/local/bin/logger

process_family() {
    local fam_name="$1"
    local family_dir="$2"

    local N=6
    for pcap in "$family_dir"/*.pcapng; do
        ((i=i%N)); ((i++==0)) && wait
        basename="$OUTPUT_DIR/$fam_name/${pcap##*/}"
        trapfile="${basename%.*}.trap"
        csvfile="${basename%.*}.csv"
        echo "Processing - $pcap -> $csvfile"
        $IPFIXPROBE_CMD -i "pcap;file=$pcap" -s "cache;s=22;active=300;inactive=65" -p "pstats;includezeroes" -p dns -o "unirec;i=f:$trapfile;p=(pstats,dns)" && $LOGGER_CMD -i "f:$trapfile" -t > "$csvfile" &
    done
}


echo "Working on dataset at $DATASET_PATH"

for family_dir in "$DATASET_PATH"/*; do
    fam_name=${family_dir%*/}
    fam_name=${fam_name##*/}
    echo "Processing family - $family_dir -> $OUTPUT_DIR/$fam_name"
    mkdir -p "$OUTPUT_DIR/$fam_name"
    process_family "$fam_name" "$family_dir"
done
