# Navigating Spatio-Temporal Long-Short Heterogeneity: A Dual-Stream Graph Neural Network for Localized Sparse Meteorological Forecasting

This it the official github for "Navigating Spatio-Temporal Long-Short Heterogeneity: A Dual-Stream Graph Neural Network for Localized Sparse Meteorological Forecasting"

## Requirements
- python>=3.8
- torch>=2.0.0
- numpy>=1.23.5
- pandas>=1.5.3
- scipy>=1.10.1
- tables>=3.8.0
- pywavelets>=1.4.1

## Usage
### Model Training
We provide default training codes in `train.py`. You can train the model as follows:
```
python train.py
```
For more parameter information, please refer to `train.py`. We provide a more detailed and complete command description for the training code:
```
python -u train.py
 --device DEVICE
 --data DATA
 --input_dim INPUT_DIM
 --channels CHANNELS
 --num_nodes NUM_NODES
 --input_len INPUT_LEN
 --output_len OUTPUT_LEN
 --batch_size BATCH_SIZE
 --learning_rate LEARNING_RATE
 --dropout DROPOUT
 --weight_decay WEIGHT_DECAY
 --epochs EPOCHS
 --print_every PRINT_EVERY
 --save SAVE
 --es_patience ES_PATIENCE
```
### Model Testing
For the testing, you can run the code below:
```
python test.py
```



