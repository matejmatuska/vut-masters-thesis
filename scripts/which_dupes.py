from collections import defaultdict
import re
import sys

# Dictionary to hold file identifier as key and families as values
file_families = defaultdict(set)

with open(sys.argv[1], "r") as file:
    for line in file:
        match = re.search(r"/extracted/(\S*)/(\d{6}-\S+\.behavioral[12])\.pcapng$", line)
        if match:
            family, file_id = match.groups()
            file_families[file_id].add(family)

# Print out file identifiers with multiple families
for file_id, families in file_families.items():
    if len(families) > 1:
        print(f"File ID: {file_id}, Families: {', '.join(families)}")

