#!/bin/bash

# stop on error
set -e

for i in {25..35}
do
  # Create a new file for each value
  new_file="voronoi-seed_${i}.slurm"
  cp voronoi-0n.slurmtemplate $new_file

  # Replace the values in the new file
  # sed -i "s/vor-0n/vor-${i}n/g" $new_file
  # sed -i "s/vor\/0/vor\/${i}/g" $new_file
  sed -i "s/seedint/${i}/g" $new_file
done