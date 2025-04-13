import os
import re
import shutil
import sys
from collections import defaultdict


def get_fams_per_sample(base_directory):
    hash_to_fams = defaultdict(set)  # hash to fams

    for root, _, files in os.walk(base_directory):
        for file in files:
            fullpath = os.path.join(root, file)
            match = re.search(r'/([^/]+)/(\d{6}-\S+\.behavioral[12])\.pcapng$', fullpath)
            if not match:
                #print(f"Ignoring filename: {fullpath}")
                continue

            family, hash = match.groups()
            hash_to_fams[hash].add(family)

    return hash_to_fams

    return
    for hash, fullpath in hash_to_fams.items():
        if fullpath != "duplicate":
            parts = fullpath.split(os.sep)
            family = parts[-2]
            if not os.path.exists(os.path.join(combined_directory, family)):
                os.makedirs(os.path.join(combined_directory, family))

            print(f"Moving unique file: {fullpath} to {os.path.join(combined_directory, family, parts[-1])}")
            shutil.move(fullpath, os.path.join(combined_directory, family, parts[-1]))



if __name__ == "__main__":
    extracted_directory = sys.argv[1]
    combined_directory = sys.argv[2]

    if not os.path.exists(combined_directory):
        os.makedirs(combined_directory)

    sample_to_fams = get_fams_per_sample(extracted_directory)

    for hash, fams in sample_to_fams.items():
        if len(fams) == 1:
            fam = fams.pop()
            famdir = os.path.join(combined_directory, fam)
            if not os.path.exists(famdir):
                os.makedirs(famdir)

            temp = os.path.join(fam, hash + ".pcapng")
            src = os.path.join(extracted_directory, temp)
            dest = os.path.join(combined_directory, temp)
            print(f"Moving unique file: {src} to {dest}")
            shutil.move(src, dest)
        else:
            print(f"Skipping duplicate file: {hash}")
