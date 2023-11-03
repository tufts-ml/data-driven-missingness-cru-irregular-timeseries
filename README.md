# mnar-cru-irregular-timeseries
Code for irregular time-series models with missing-not-at-random assumption

## Run toy experiments
Run the following lines of code for reproducing the toy experiments

### MNAR-CRU
`python run_toy_experiment_cru.py --random_seed 35 --mnar True`

### CRU 
`python run_toy_experiment_cru.py --random_seed 35 --mnar False`

![toy_experiments](https://github.com/tufts-ml/mnar-cru-irregular-timeseries/blob/main/toydata_extrapolation.png)
