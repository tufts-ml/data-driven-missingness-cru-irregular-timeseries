# mnar-cru-irregular-timeseries
Code for irregular time-series models with missing-not-at-random assumption

## Run toy experiments
Run the following lines of code for reproducing the toy experiments

### MNAR-CRU
`python CRU/run_toy_experiment_cru.py --random_seed 35 --mnar True`

### CRU 
`python CRU/run_toy_experiment_cru.py --random_seed 35 --mnar False`

### mTAND 
`bash mTAND/run_toy_mnar_extrapolation.sh`


### pVAE 
`bash pVAE/run_toy_mnar_extrapolation.sh`

![toy_experiments](https://github.com/tufts-ml/mnar-cru-irregular-timeseries/blob/main/toydata_extrapolation.png)
