
# eICU data preprocessing

## Workflow

### Extracting data from eICU 
Ensure you have access to Physionet and download the files from the eICU Collaborative Database
[eICU Collaborative Database](https://physionet.org/content/eicu-crd/2.0/) using the following command

`>> wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/eicu-crd/2.0/`

### Pre-process the chart events to extract relevant vitals and labs
Run the "make_csv_dataset_from_raw.py" script to generate 2 files :

 - features_per_tstep.csv.gz (downsampled vitals with timestamps)
 - outcomes_per_seq.csv (outcomes)

`>> python make_csv_dataset_from_raw.py`

### Convert raw data into format for downstream models and split into train/valid/test
Run all the cells in the "create_irregular_ts_dataset.ipynb" notebooks to generate the train/valid/test splits 

Note : The train valid test files will be saved in the "data/classifier_train_test_split_dir" folder


 