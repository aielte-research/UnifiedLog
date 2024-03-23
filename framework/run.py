import yaml
from yaml.loader import SafeLoader
import neptune as neptune
import os
import torch
import pickle
from models.transformer_encoder import TransformerEncoder
from models.anomaly_detector import AnomalyDetector
import copy
import argparse
from dotenv import load_dotenv
load_dotenv()

def initialize_neptune():
    print("Initializing neptune logging...")
    try:
        api_token = os.environ["NEPTUNE_API_TOKEN"]
    except:
        raise ValueError("NEPTUNE_API_TOKEN not set as environment variable")
    run = neptune.init_run(
        project="aielte/CyberML",
        api_token=api_token,
    ) 
    return run


def dict_to_neptune_loggable(dict):
    for key in dict.keys():
        try:
            dict_to_neptune_loggable(dict[key])
        except:
            dict[key] = str(dict[key])
    return dict


def handle_encoder_training(transformer_encoder, params):
    ### DIRECOTRY GIVEN IN CONF
    if type(params["transformer_encoder"]["train_paths"]) == str:
        print(f'Loading train data for encoder... (Loaded {len(os.listdir(params["transformer_encoder"]["train_paths"]))} files)')    
        train_paths = []
        for file in os.listdir(params["transformer_encoder"]["train_paths"]):
            train_fpath = os.path.join(params["transformer_encoder"]["train_paths"], file)
            train_paths.append(train_fpath)
    ### LIST OF FILES GIVEN ON CONF
    else:
        print(f"Loading train data for encoder... (Loaded {len(params['transformer_encoder']['train_paths'])} files)")    
        train_paths = params["transformer_encoder"]["train_paths"]

    transformer_encoder_train_data_dict = {}
    for i in range(len(train_paths)):
        with open(train_paths[i], "rb") as file:
            transformer_encoder_train_data = pickle.load(file)
        log_name = ".".join(train_paths[i].split("/")[-1].split(".")[:-1])
        transformer_encoder_train_data_dict[log_name] = [torch.Tensor(x).long() for x in transformer_encoder_train_data]


    # Reduce maximum number of lines per dataset
    if params["transformer_encoder"]["max_train_data_size"] is not None:
        print(f'Keeping the first {params["transformer_encoder"]["max_train_data_size"]} lines from each dataset.')
        for key in transformer_encoder_train_data_dict.keys():
            transformer_encoder_train_data_dict[key] = transformer_encoder_train_data_dict[key][:int(params["transformer_encoder"]["max_train_data_size"])]


    transformer_encoder.train(
        data = transformer_encoder_train_data_dict, 
        train_val_test_split = params["transformer_encoder"]["train_val_test_split"], 
        batch_size = params["transformer_encoder"]["batch_size"], 
        lr = params["transformer_encoder"]["lr"], 
        save_path = params["transformer_encoder"]["save_path"], 
        epochs = params["transformer_encoder"]["epochs"],
        padding_value = params["transformer_encoder"]["pad_token_id"],
        save_every_epoch=params["transformer_encoder"]["save_every_epoch"]
    )


def generate_openstack_labels(anomaly_detector_data_dict, anomaly_detector_labels_dict):
    for key in anomaly_detector_data_dict.keys():
        if "openstack" in key:
            if "abnormal" not in key:
                anomaly_detector_labels_dict[key] = torch.zeros(len(anomaly_detector_data_dict[key]))
            else:
                anomaly_detector_labels_dict[key] = torch.ones(len(anomaly_detector_data_dict[key]))


def load_data_detector(params, train=True):
    if train:
        path_param = "train_paths"
    else:
        path_param = "test_data_paths"
    ### DIRECOTRY GIVEN IN CONF
    if type(params["anomaly_detector"][path_param]) == str:
        print(f'Loading train data for encoder... (Loaded {len(os.listdir(params["anomaly_detector"][path_param]))} files)')    
        paths = []
        for file in os.listdir(params["anomaly_detector"][path_param]):
            train_fpath = os.path.join(params["anomaly_detector"][path_param], file)
            paths.append(train_fpath)
    ### LIST OF FILES GIVEN ON CONF
    else:
        print(f"Loading train data for encoder... (Loaded {len(params['anomaly_detector'][path_param])} files)")    
        paths = params["anomaly_detector"][path_param]

    print(f'Loading train data for detector...(Loaded {len(paths)} files)')

    anomaly_detector_train_data_dict = {}
    for i in range(len(paths)):
        with open(paths[i], "rb") as file:
            anomaly_detector_train_data = pickle.load(file)
        log_name = ".".join(paths[i].split("/")[-1].split(".")[:-1])

        if "HDFS_1" in log_name or "hadoop" in log_name.lower():
            if "HDFS_1" in log_name:
                log_name = "HDFS_1"
            elif "hadoop" in log_name.lower():
                log_name = "hadoop"
            anomaly_detector_train_data_dict[log_name] = {}
            for key in anomaly_detector_train_data.keys():
                anomaly_detector_train_data_dict[log_name][key] = [torch.Tensor(x).long().to("cuda") for x in anomaly_detector_train_data[key]]
        else:
            anomaly_detector_train_data_dict[log_name] = [torch.Tensor(x).long().to("cuda") for x in anomaly_detector_train_data]
    
    for key in anomaly_detector_train_data_dict.keys():
        print(f"Dataset {key} has \t\t {len(anomaly_detector_train_data_dict[key])} lines.")
    
    return anomaly_detector_train_data_dict


def load_labels_detector(anomaly_detector_data_dict, params, train=True):
    if train:
        label_path_param = "label_paths"
        data_type_str = "train"
    else:
        label_path_param = "test_labels"
        data_type_str = "test"
    ### DIRECOTRY GIVEN IN CONF
    if type(params["anomaly_detector"][label_path_param]) == str:     
        print(f'Loading {data_type_str} data for detector... (Loaded {len(os.listdir(params["anomaly_detector"][label_path_param]))} files)')    
        paths = []
        for file in os.listdir(params["anomaly_detector"][label_path_param]):
            fpath = os.path.join(params["anomaly_detector"][label_path_param], file)
            paths.append(fpath)
    ### LIST OF FILES GIVEN ON CONF
    else:
        print(f"Loading {data_type_str} data for detector... (Loaded {len(params['anomaly_detector'][label_path_param])} files)")    
        paths = params["anomaly_detector"][label_path_param]

    anomaly_detector_train_labels_dict = {}
    for i in range(len(paths)):
        if "openstack" in paths[i]:
            continue

        with open(paths[i], "rb") as file:
            anomaly_detector_train_labels = pickle.load(file)

        log_name = ".".join(paths[i].split("/")[-1].split(".")[:-1])
        if "HDFS_1" in log_name:
            log_name = "HDFS_1"
            anomaly_detector_train_labels_dict[log_name] = {}
            for line in anomaly_detector_train_labels:
                block = line.split(",")[0].strip()
                if line.split(",")[-1].strip() == "Normal":
                    label = 0
                else: 
                    label = 1
                anomaly_detector_train_labels_dict[log_name][block] = label
        elif "hadoop" in log_name.lower():
            log_name = "hadoop"
            anomaly_detector_train_labels_dict[log_name] = {}
            block_label = {}
            for line in anomaly_detector_train_labels:
                block = line.split(",")[0].strip()
                if line.split(",")[-1].strip() == "Normal":
                    block_label[block] = 0
                else: 
                    block_label[block] = 1
            for container in anomaly_detector_data_dict[log_name].keys():
                application_substing = "_".join(container.split("_")[:3])
                anomaly_detector_train_labels_dict[log_name][container] = block_label[application_substing]
        else:
            # BGL, Thunderbird
            anomaly_detector_train_labels_dict[log_name] = torch.Tensor(anomaly_detector_train_labels)
    return anomaly_detector_train_labels_dict


def encode_data(transformer_encoder, data, batch_size, pad_token_id):
    encoded = transformer_encoder.encode(
        data=data,
        batch_size = batch_size, 
        padding_value = pad_token_id
    )
    return encoded


def main(conf_path):
    ###################
    ### LOAD PARAMS ###
    ###################

    with open(conf_path) as f:
        params = yaml.load(f, Loader=SafeLoader)
    if params["neptune_logging"]:
        run = initialize_neptune()
        run["params"] = dict_to_neptune_loggable(copy.deepcopy(params))
    else:
        run = None

    ####################
    ### ENCODER PART ###
    ####################

    transformer_encoder = TransformerEncoder(
        run=run,
        load_path=params["transformer_encoder"]["load_path"],
        mask_ignore_token_ids = params["transformer_encoder"], 
        mask_token_id = params["transformer_encoder"]["mask_token_id"], 
        pad_token_id = params["transformer_encoder"]["pad_token_id"], 
        mask_prob = params["transformer_encoder"]["mask_prob"], 
        replace_prob = params["transformer_encoder"]["replace_prob"], 
        num_tokens = params["transformer_encoder"]["num_tokens"], 
        max_seq_len = params["transformer_encoder"]["max_seq_len"], 
        attn_layers_dim = params["transformer_encoder"]["attn_layers"]["dim"], 
        attn_layers_depth = params["transformer_encoder"]["attn_layers"]["depth"], 
        attn_layers_heads = params["transformer_encoder"]["attn_layers"]["heads"],
    )

    # Encoder training
    if params["transformer_encoder"]["epochs"] > 0:
        handle_encoder_training(transformer_encoder=transformer_encoder, params=params)

    #####################
    ### DETECTOR PART ###
    #####################

    # Load data for detector
    anomaly_detector_train_data_dict = load_data_detector(params=params, train=True)

    # Load labels for detector
    anomaly_detector_train_labels_dict = load_labels_detector(anomaly_detector_data_dict=anomaly_detector_train_data_dict, params=params, train=True)

    # Openstack label generation
    generate_openstack_labels(anomaly_detector_data_dict=anomaly_detector_train_data_dict, anomaly_detector_labels_dict=anomaly_detector_train_labels_dict)


    ### Loading test data
    anomaly_detector_test_data_dict = load_data_detector(params=params, train=False)
    anomaly_detector_test_labels_dict = load_labels_detector(anomaly_detector_data_dict=anomaly_detector_test_data_dict, params=params, train=False)

    # Openstack label generation
    generate_openstack_labels(anomaly_detector_data_dict=anomaly_detector_test_data_dict, anomaly_detector_labels_dict=anomaly_detector_test_labels_dict)

    # Encode loaded data with encoder
    encoded_train = encode_data(
        transformer_encoder=transformer_encoder, 
        data=anomaly_detector_train_data_dict, 
        batch_size=params["transformer_encoder"]["batch_size"],
        pad_token_id=params["transformer_encoder"]["pad_token_id"]
    )
    

    anomaly_detector = AnomalyDetector(
        run=run,
        load_path=params["anomaly_detector"]["load_path"],
        embed_dim = params["transformer_encoder"]["attn_layers"]["dim"], 
        ff_dim = params["anomaly_detector"]["ff_dim"], 
        max_len = params["anomaly_detector"]["max_len"], 
        num_heads = params["anomaly_detector"]["num_heads"], 
        dropout = params["anomaly_detector"]["dropout"],
    )

    anomaly_detector.train(
        X=encoded_train, 
        Y=anomaly_detector_train_labels_dict, 
        train_val_test_split = params["anomaly_detector"]["train_val_test_split"], 
        save_path = params["anomaly_detector"]["save_path"], 
        lr = params["anomaly_detector"]["lr"], 
        lr_decay_step_size = params["anomaly_detector"]["lr_decay_step_size"], 
        lr_decay_gamma = params["anomaly_detector"]["lr_decay_gamma"], 
        early_stop_tolerance = params["anomaly_detector"]["early_stop_tolerance"], 
        early_stop_min_delta = params["anomaly_detector"]["early_stop_min_delta"], 
        batch_size = params["anomaly_detector"]["batch_size"], 
        epochs = params["anomaly_detector"]["epochs"],
        balancing_ratio=params["anomaly_detector"]["balancing_ratio"]
    )

    encoded_test = encode_data(
        transformer_encoder=transformer_encoder, 
        data=anomaly_detector_test_data_dict, 
        batch_size=params["transformer_encoder"]["batch_size"],
        pad_token_id=params["transformer_encoder"]["pad_token_id"]
    )

    anomaly_detector.eval(
        batch_size=params["anomaly_detector"]["batch_size"], 
        X_test=encoded_test,
        Y_test=anomaly_detector_test_labels_dict
    )

    if run is not None:
        run.stop()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("No cuda device found! Exiting...")
        raise Exception("NO CUDA ERROR")
    
    ####################
    ### PARSE PARAMS ###
    ####################

    parser = argparse.ArgumentParser(
                    prog='UnifiedLog',
                    description='This script trains an ecoder on multiple datasets, and then trains an anomaly detector.',
                    )

    parser.add_argument('-c', '--config')       # option that takes a value
    parser.add_argument('-t', '--threads', default=20)
    args = parser.parse_args()
    torch.set_num_threads(int(args.threads))

    if args.config.endswith(".yaml"):
        main(conf_path=args.config)
    else:
        for conf in os.listdir(args.config):
            main(conf_path=os.path.join(args.config, conf))
    
    