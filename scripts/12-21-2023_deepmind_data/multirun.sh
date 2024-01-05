#!/bin/bash

# Default values
min_max_atoms=14
max_max_atoms=20
version_number="v1"
increment=1  # Default increment

# Parse command line options
while getopts "m:n:v:i:" opt; do
  case $opt in
    m) min_max_atoms=$OPTARG ;;
    n) max_max_atoms=$OPTARG ;;
    v) version_number=$OPTARG ;;
    i) increment=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Main loop with specified increment
for max_atoms in $(seq $min_max_atoms $increment $max_max_atoms); do
    echo "Launching job for max_atoms = $max_atoms"
    /home/gridsan/tmackey/cdvae/train_launcher.sh mp_20_dm "ogCDVAE_dm_${max_atoms}u${version_number}" "ogCDVAE_dm_${max_atoms}u${version_number}" $max_atoms
done