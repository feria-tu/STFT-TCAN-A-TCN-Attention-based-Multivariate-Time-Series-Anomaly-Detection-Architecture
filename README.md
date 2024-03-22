# STFT-TCAN-A-TCN-Attention-based-Multivariate-Time-Series-Anomaly-Detection-Architecture
STFT-TCAN: A TCN-Attention based Multivariate Time Series Anomaly Detection Architecture with Time-Frequency Analysis

## INSTALLATION
```bash
conda create -n stft python=3.8
conda activate stft
pip3 install -r requirements.txt
```

## Datasets
you can get the dataset through the links below:
```
MBA: 
https://physionet.org/content/mitdb/1.0.0/

SMAP_MSL: 
data taken from:
https://s3-us-west-2.amazonaws.com/telemanom/data.zip
labeled anomalies from:
https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv

SMD:
This dataset is taken as is from https://github.com/NetManAIOps/OmniAnomaly

SWaT:
series.json taken from : https://raw.githubusercontent.com/JulienAu/Anomaly_Detection_Tuto/master/Data/serie2.json

WADI:
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#wadi
```

## Dataset Preprocessing

```bash
python3 data_prepocessed.py
```

## RUN

```bash
python3 main.py --dataset <dataset>
```
dataset: SMD\SWaT\MSL\MBA\WADI\SMAP
