#!/bin/bash
for i in {3..10}
do
  # Create a new file for each value
  new_file="milling-${i}n.slurm"
  cp milling-0n.slurmtemplate $new_file

  # Replace the values in the new file
  sed -i "s/mill-0n/mill-${i}n/g" $new_file
  sed -i "s/mill\/0/mill\/${i}/g" $new_file
  sed -i "s/-N 0/-N ${i}/g" $new_file
done