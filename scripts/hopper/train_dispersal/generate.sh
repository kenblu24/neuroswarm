#!/bin/bash

# stop on error
set -e

for i in {25..35}
do
  # Create a new file for each value
  new_file="dispersal-seed_${i}.slurm"
  cp dispersal-0n.slurmtemplate $new_file

  # Replace the values in the new file
  # sed -i "s/dis-0n/dis-${i}n/g" $new_file
  # sed -i "s/dis\/0/dis\/${i}/g" $new_file
  sed -i "s/seedint/${i}/g" $new_file
done