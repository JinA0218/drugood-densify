import numpy as np
import os

folder_path = 'data/perms'

# List all .npy files in the folder
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# Load and print each .npy file
for file_name in npy_files:
    full_path = os.path.join(folder_path, file_name)
    data = np.load(full_path)
    print(f"Contents of {file_name}:\n{data}\n")
