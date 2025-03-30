#!/bin/bash

# stop on error
set -e

for i in {25..35}
do
  # Create a new file for each value
  new_file="milling-seed_${i}.slurm"
  cp milling-0n.slurmtemplate $new_file

  # Replace the values in the new file
  # sed -i "s/mill-0n/mill-${i}n/g" $new_file
  # sed -i "s/mill\/0/mill\/${i}/g" $new_file
  sed -i "s/seedint/${i}/g" $new_file
done