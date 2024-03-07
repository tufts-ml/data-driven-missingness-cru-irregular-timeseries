#!/usr/bin/env bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 1-02:00          # Runtime in D-HH:MM
#SBATCH -p ccgpu            # Partition to submit to
#SBATCH --mem-per-cpu 20000  # Memory (in MB) per cpu
#SBATCH -o mimic_logs_dir/log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e mimic_logs_dir/log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Load the right conda environment
source activate mnar-cru

# If some ENV variables are not defined, set them to defaults
if [[ -z $random_seed ]]; then
    random_seed=0;
fi
if [[ -z $lr ]]; then
    lr=0.001;
fi
if [[ -z $mnar ]]; then
    mnar=False;
fi
if [[ -z $lsd ]]; then
    lsd=10;
fi

# Pass along all ENV variables as arguments to my Python script
python run_mimic_extrapolation.py --dataset mimic --task extrapolation -lsd $lsd --ts 0.2 --enc-var-activation square --dec-var-activation exp --trans-var-activation relu --grad-clip --num-basis 20 --bandwidth 5 --cut-time 24 --random-seed $random_seed --mnar $mnar

source deactivate