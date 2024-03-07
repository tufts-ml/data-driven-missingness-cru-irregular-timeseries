# mnar-cru-irregular-timeseries
Code for irregular time-series models with missing-not-at-random assumption

## Cloning the anonymous repo 
Please follow the instructions [here](https://github.com/fedebotu/clone-anonymous-github) to clone an anonymous repo. (Credit : Clone Anonymous Github created by fedebotu)

## Run toy experiments
Run the following lines of code for reproducing the toy experiments

### Create the environment
`conda create -n mnar-cru python=3.9.7`

`conda activate mnar-cru`

`pip install -r requirements.txt`

### MNAR-CRU
`python CRU/run_toy_experiment_cru.py --random_seed 68 --mnar True`

### CRU 
`python CRU/run_toy_experiment_cru.py --random_seed 68 --mnar False`

- The results will be saved in the "CRU/training_results/toy_mnar/test_true_vs_predicted\*mnar=True/False\*.png"
> Please let the CRU model run for atleast 100 epochs (default). It should take no more than 8-10 minutes on any machine. 

### mTAND 
`bash mTAND/run_toy_mnar_extrapolation.sh`
- The results will be saved in mTAND/results folder

### LatentODE
`bash LatentODE/run_toy_mnar_experiment.sh`
- The results will be saved in LatentODE/results folder

### NeuralCDE
`bash NeuralCDE/run_toy_mnar_experiment.sh`
- The results will be saved in NeuralCDE/results folder

### pVAE 
`bash pVAE/run_toy_mnar_extrapolation.sh`
- The results will be saved in NeuralCDE/results folder

> Note : To run the pVAE experiment, please create a separate enviroment using this [requirements.txt](https://github.com/steveli/partial-encoder-decoder/blob/master/requirements.txt) file.

![toy_experiments](toydata_extrapolation.png)

## Run MIMIC-IV experiments
-Follow the instructions to pre-process the data in [data_preprocessing/MIMIC-IV](data_preprocessing/MIMIC-IV)

Then run the script to train MNAR-CRU and CRU
`bash CRU/launch_cru_extrapolation_mimic.sh run_here`

## Run eICU experiments
-Follow the instructions to pre-process the data in [data_preprocessing/eICU](data_preprocessing/eICU)

Then run the script to train MNAR-CRU and CRU
`bash CRU/launch_cru_extrapolation_eicu.sh run_here`
