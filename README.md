# UnifiedLog

## Project description
Our research aimed to develop a log anomaly detector that can be used effectively immediately on new logs.
We propose <b>UnifiedLog</b>, a log anomaly detection framework that consists of two parts: a transformer based <b>encoder</b> for a general log representation, that utilizes only the semantic information in logs and a transformer based <b>detector</b> model which is capable of anomaly detection in sequences of representations.
Since we use an approach that allows us to represent all types of logs in a single representation space, we can build an anomaly detection model to detect anomalies on multiple and different datasets.
To train the encoder part of UnifiedLog we use masked language modelling (MLM), a self-supervised training method to learn a unified representation for any line of log.
A general log representation approach must utilize unlabeled data as the availability of labeled datasets is not satisfying in this field.
The detector part of UnifiedLog is inspired by NeuralLog (https://github.com/LogIntelligence/NeuralLog), which is a transformer based anomaly detector designed for the classification task on a sequence of representations.
By unified representations, the detector can be trained on multiple datasets simultaneously, improving predictive capabilities by the cross-domain information.
We experiment with varying the training datasets of both the encoder and detector part of UnifiedLog to show that they can generalize to unseen datasets.
This is the first comprehensive model to detect anomalies on datasets it has never been trained on. 


To summarize our main contributions are as follows:
- We propose UnifiedLog, a framework capable of log anomaly detection on multiple datasets simultaneously. 
- UnifiedLog is the first model that aims to predict anomalies on datasets not used in training.
- We confirm that instead of log parsing to represent raw log messages, it is better to use a single unified language model.
- We suggest a new approach to evaluating log anomaly detection systems by combining performance metrics on multiple 


## Requirements
To replicate our results using the UnifiedLog framework, we suggest the following hardware specifications:
- <b>GPU:</b> NVIDIA A100 with 80GB of VRAM
- <b>Storage:</b> At least 200GB of available disk space
- <b>RAM:</b> 50GB or more

## Installation:
```
conda env create -f environment.yml
```

## Download the datasets available on Loghub with <i>loghub_downloader.py</i>.

```
python3 loghub_downloader.py -s <save-folder>
```

## Preprocess datasets with <i>data_preprocess.py</i>.

```
python3 data_preproceess.py -d <path-to-downloaded-logs> -s <save-folder> -l <maximum-lines-per-dataset> -v <num-of-tokens> -a <ASCII-policy> -n <number-policy>
```
The script removes or replaces non-ASCII characters with a special token based on the ASCII-policy, then either replaces all numeric characters with a combined [NUM] token or flags the 0-9 characters as special tokens.
The script then trains a Wordpiece tokenizer on all the datasets and saves the tokenized versions of them. 

Three folders are created by this scipt: 
- <i>tokenized<i/>
- <i>tokenized_for_detector<i/>
- <i>labels<i/>.

## To train and evaluate UnifiedLog run the <i>run.py</i> script.

```
python3 run.py -c <conf-file> -t <cpu-threads>
```

### Example config file:
```
name: example # Name of the run in neptune
neptune_logging: false # Export NEPTUNE_API_KEY as environment variable if set to true
transformer_encoder:
  train_paths: "path_to/tokenized/" # Folder containing data tokenized for the encoder (also accepts a list directly containing files)
  load_path: "path_to/saved_model" # Load encoder from previous save
  save_path: "path_to/model_name" # Save path of the encoder
  save_every_epoch: True # If True a model with save_path + _epoch_n.pkl will be saved every epoch
  train_val_test_split: [0.8, 0.9]
  mask_prob: 0.15
  replace_prob: 0.9
  num_tokens: 1004
  max_seq_len: 128
  attn_layers:
    dim: 16 # This affects the detector part's embedding also
    depth: 4
    heads: 6
  batch_size: 4096
  lr: 0.00003
  epochs: 5
  mask_token_id: 1002
  pad_token_id: 1003
  max_train_data_size: 10000000      # Cap maximum lines used from one dataset for training
anomaly_detector:
  train_paths: "path_to/tokenized_for_detector" # folder created by data_preprocess.py
  label_paths: "path_to/labels" # folder created by data_preprocess.py
  test_data_paths: "path_to/tokenized_for_detector"
  test_labels: "path_to/labels" 
  load_path: null # Load detector from previous save
  save_path: null # Save path of the detector
  train_val_test_split: [0.8, 0.9] 
  lr_decay_step_size: 25
  lr_decay_gamma: 0.9
  early_stop_tolerance: 3
  early_stop_min_delta: 0
  batch_size: 64
  epochs: 200
  embed_dim: 64
  ff_dim: 256
  max_len: 20
  num_heads: 8
  dropout: 0.5
  lr: 0.00003
  balancing_ratio: 1
```

## Citation
If you use this code in your research, please cite the corresponding paper:

Instert Citation Here

## Contributors

- Lajos Muzsai (muzsailajos@protonmail.com)
- András Lukács (andras.lukacs@ttk.elte.hu)

## License 
This project is licensed under the MIT License - see the LICENSE file for details.
