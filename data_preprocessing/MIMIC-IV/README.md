
# MIMIC-IV in-ICU data preprocessing

## Workflow

### Downloading the chart events and ICU mortality outcomes
Clone the [MIMIC-IV Data-Pipeline](https://github.com/healthylaife/mimic-iv-data-pipeline) repo and follow the instructions to extract the chart events and ICU mortality outomes. We used the "data extraction" in the [mainPipeline.ipynb](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/mainPipeline.ipynb) notebook to extract the chart events without any disease filters.

Running the notebook generates 3 files : 

 - preproc_chart.csv.gz (all the vitals with timestamps)
 - d_items.csv.gz (file linking the vital IDs to the vital names)
 - cohort_icu_mortality.csv.gz (file containing in-ICU mortality outcome per admission)

Make sure to save these files in the "data" folder

### Pre-process the chart events to extract relevant vitals and labs
Run the "make_csv_dataset_from_raw.py" script to generate 3 files :

 - features_per_tstep.csv.gz (downsampled vitals with timestamps)
 - outcomes_per_seq.csv (outcomes)
 - demographics.csv.gz (demographics per admission)

`>> python make_csv_dataset_from_raw.py`

### Convert raw data into format for downstream models and split into train/valid/test
Run all the cells in the "create_irregular_ts_dataset.ipynb" notebooks to generate the train/valid/test splits 

Note : The train valid test files will be saved in the "data/classifier_train_test_split_dir" folder


 