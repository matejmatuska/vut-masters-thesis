"""
This script is used to find duplicate PCAPNG files in a family directory based on the
hash in the filename.
"""

import os
import sys

if len(sys.argv) < 2:
    print("Usage: python find_dupes.py <directory>")
    sys.exit(1)

seen_hashes = set()

for filename in os.listdir(sys.argv[1]):
    if filename.endswith(".pcapng"):
        parts = filename.split('-')
        if len(parts) > 1:
            # keep the behavioralX part of the filename
            hash = parts[1].rstrip('.pcapng')

            # print(f"Hash: {hash}")
            if hash in seen_hashes:
                print(f"Duplicate: {filename}")
                # print(f"Removing duplicate: {filename}")
                # os.remove(os.path.join(sys.argv[1], filename))
            else:
                seen_hashes.add(hash)
        else:
            print(f"Invalid filename: {filename}")
