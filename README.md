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

### mTAND 
`bash mTAND/run_toy_mnar_extrapolation.sh`

### pVAE 
`bash pVAE/run_toy_mnar_extrapolation.sh`
Note : To run the pVAE experiment, please create a separate enviroment using this [requirements.txt](https://github.com/steveli/partial-encoder-decoder/blob/master/requirements.txt) file.


![toy_experiments](https://github.com/tufts-ml/mnar-cru-irregular-timeseries/blob/main/toydata_extrapolation.png)
