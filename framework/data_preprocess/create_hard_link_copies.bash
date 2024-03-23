#!/bin/bash

# Source directory containing .pkl files
source_dir="preprocessed_NUM_5m/tokenized/"

# Destination directory for hard links
dest_dir="tokenized_no_thunderbird/"
# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Loop through all .pkl files in the source directory
for file in "$source_dir"*.pkl; do
    # Extract the filename without the path
    filename=$(basename "$file")

    # Create hard link in the destination directory
    ln "$file" "$dest_dir$link_prefix$filename"
done
