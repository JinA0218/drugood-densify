import re
from collections import defaultdict

# Path to your input file
file_path = input('your_file.txt: ')

# Read the whole file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Group blocks
blocks = []
current_block = []

for line in lines:
    if re.match(r'^\w+ (bit|count) lr:', line.strip()):  # Start of a new block
        if current_block:
            blocks.append(current_block)
        current_block = [line]
    else:
        if line.strip() != '':
            current_block.append(line)

# Add last block
if current_block:
    blocks.append(current_block)

# Parse each block
grouped_blocks = defaultdict(list)

for block in blocks:
    header = block[0].strip()
    match = re.match(r'^(\w+ \w+)', header)
    if match:
        group_name = match.group(1)  # e.g., 'hivprot bit' or 'dpp4 count'
    else:
        continue

    # Extract Ncd list length
    ncd_line = next((l for l in block if 'Ncd' in l), None)
    if ncd_line:
        ncd_list = eval(re.search(r'Ncd\s*:\s*(\[.*?\])', ncd_line).group(1))
        ncd_len = len(ncd_list)
    else:
        ncd_len = 0

    grouped_blocks[group_name].append((ncd_len, block))

# Sort within each group by Ncd length (descending)
for group in grouped_blocks:
    grouped_blocks[group] = sorted(grouped_blocks[group], key=lambda x: -x[0])

# Print the sorted results
for group, entries in grouped_blocks.items():
    print(f'\n=== {group} ===\n')
    for ncd_len, block in entries:
        for line in block:
            print(line.strip())
        print()  # blank line between blocks
