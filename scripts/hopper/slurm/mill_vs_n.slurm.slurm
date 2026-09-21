#!/bin/bash



## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=mill_vs_n_50

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=normal

## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file

## Slurm can send you updates via email
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=16G      # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=1-12:00  # Total time needed for job: Days-Hours:Minutes

#SBATCH --cpus-per-task 64


## Load the relevant modules needed for the job
source ~/neuromorphic/turtwig/scripts/hopper/neuromodules.sh
cd ~/neuromorphic/turtwig

export MAKE_PARENTS=true
export PROJ_EXIST_OKAY=true

## Run your program or script
python mill_n_vs_seed.py --rng_seed 2026 --Nrange 3-50 --trials 100 --force
