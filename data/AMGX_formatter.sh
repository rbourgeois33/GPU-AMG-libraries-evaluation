#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <aij_file> <rhs_file>"
    exit 1
fi

# Input files
aij_file="$1"
rhs_file="$2"

# Output file
output_file="AMGX_system.mtx"

# Clear output file or create it
> "$output_file"

# Extract and write header from aij file
grep '^%%' "$aij_file" >> "$output_file"

# Add AMGX header
echo "%%AMGX rhs" >> "$output_file"

# Append the rest of aij file (excluding its header)
grep -v '^%%' "$aij_file" >> "$output_file"

# Append rhs file without its header (assuming header lines start with '%')
grep -v '^%' "$rhs_file" >> "$output_file"

echo "AMGX formatted file created: $output_file"