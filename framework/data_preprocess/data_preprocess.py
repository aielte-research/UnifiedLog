import argparse
import os
import pandas as pd
import string
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import pickle
import numpy as np


def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )


def replace_non_ascii(x):
    return ''.join('[NONA]' if char not in string.printable else char for char in x)


def replace_num(x):
    for i in range(10):
        x = x.replace(str(i), "[NUM]")
    return x


def read_file_contents(file, max_log_lines, save_path, verbose=True):
    def read_lines_from_file(file, data, encoding):
        with open(file, encoding=encoding) as f:
            labels = []
            if "Thunderbird" in file or "BGL" in file:
                track_labels = True
            else:
                track_labels = False
            idx = 0
            for line in f:
                idx += 1
                if line[0] == "-":
                    line = line[1:]
                    if track_labels:
                        labels.append(1)
                elif track_labels and len(line.strip()) > 0:
                    labels.append(0)
                if len(line.strip()) > 0:
                    data.append(line.strip())
                if idx == max_log_lines:
                    break
            if track_labels:
                if not os.path.exists(os.path.join(save_path, "labels")):
                    os.makedirs(os.path.join(save_path, "labels"))
                with open(os.path.join(save_path, f"labels/{file.split('/')[-1].split('.')[0]}.pkl"), "wb") as save_file:
                    pickle.dump(labels, save_file)
    
    try:
        data = [] 
        if verbose:
            print(f"Loading (encoding=utf-8) {file}")
        read_lines_from_file(file=file, data=data, encoding="utf-8")
    except:
        data = [] 
        if verbose:
            print(f"Loading (encoding=cp1252) {file}")
        read_lines_from_file(file=file, data=data, encoding="cp1252")
    if verbose:
        print(f"Loaded {len(data)} number of lines.")
    return data


def train_tokenizer(vocab_size, save_folder, data_df, ascii_policy, num_policy):
    data_df[:int(len(data_df)*0.8)].to_csv(os.path.join(save_folder, "./sample_logs_train.txt"), escapechar='\\', header=False, index=False)
    data_df[int(len(data_df)*0.8):int(len(data_df)*0.9)].to_csv(os.path.join(save_folder, "./sample_logs_val.txt"), escapechar='\\', header=False, index=False)
    data_df[int(len(data_df)*0.9):].to_csv(os.path.join(save_folder, "./sample_logs_test.txt"), header=False, escapechar='\\', index=False)

    files = [os.path.join(save_folder, f"sample_logs_{split}.txt") for split in ["test", "train", "val"]]

    print(f"========Using vocabulary from {files}=======")

    unk_token = "[UNK]"  # token for unknown words
    spl_tokens = ["[CLS]","[SEP]", "[UNK]"]

    if num_policy == "num_special_char":
        spl_tokens.append("[NUM]")
    elif num_policy == "0_9_special_char":
        for i in range(10):
            spl_tokens.append(str(i))
            
    if ascii_policy == "special_char":
        spl_tokens.append("[NONA]")

    tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
    trainer = WordPieceTrainer(
        vocab_size=vocab_size, 
        special_tokens=spl_tokens,
        show_progress=True
    )
    tokenizer.pre_tokenizer = Whitespace()
    
    tokenizer.train(files, trainer) # training the tokenzier
    tokenizer.save(os.path.join(save_folder, "./tokenizer-trained.json"))
    tokenizer = Tokenizer.from_file(os.path.join(save_folder, "./tokenizer-trained.json"))
    return tokenizer


def get_data_to_tokenize(data_folder, max_log_lines, save_path):
    data_to_tokenize = {}
    for file in os.listdir(data_folder):
        inner_file = os.path.join(data_folder, file)
        if os.path.isdir(inner_file):
            if inner_file.endswith("Hadoop") or inner_file.endswith("Spark") or inner_file.endswith("HDFS_v2"):
                print(f"Loading {inner_file}")
                data_to_tokenize[file] = []
                for inner_inner_file in os.listdir(inner_file):
                    if os.path.isdir(os.path.join(inner_file, inner_inner_file)):
                        print(f"\t/{inner_inner_file}")
                        inner_inner_folder = os.path.join(inner_file, inner_inner_file)
                        for logfile in os.listdir(inner_inner_folder):
                            data_to_tokenize[file] = data_to_tokenize[file] + read_file_contents(os.path.join(inner_inner_folder, logfile), max_log_lines=max_log_lines, verbose=False, save_path=save_path)
                    if len(data_to_tokenize[file]) > max_log_lines:
                        data_to_tokenize[file] = data_to_tokenize[file][:max_log_lines]
                        break
            else:
                for inner_inner_file in os.listdir(inner_file):
                    if os.path.join(inner_file, inner_inner_file).endswith(".log"):
                        data_to_tokenize[inner_inner_file.split(".")[0]] = read_file_contents(os.path.join(inner_file, inner_inner_file), max_log_lines=max_log_lines, save_path=save_path)
    print(f"Number of lines of logs to tokenize: {sum(len(x) for x in data_to_tokenize.values())}.")
    return data_to_tokenize


def tokenize_files(files_dict, save_folder, trained_tokenizer):
    if not os.path.exists(os.path.join(save_folder, f"tokenized")):
        os.makedirs(os.path.join(save_folder, f"tokenized"))
    for key in files_dict.keys():
        print(f"Tokenizing {key}")
        encoded = []
        for line in files_dict[key]:
            if line[0] == "-":
                line = line[1:]
            if len(line.strip()) > 0:
                encoded.append(trained_tokenizer.encode("[CLS]"+line+"[SEP]").ids)
        with open(os.path.join(save_folder, f"tokenized/{key}.pkl"), "wb") as save_file:
            pickle.dump(encoded, save_file)

        if not os.path.exists(os.path.join(save_folder, f"tokenized_for_detector")):
            os.makedirs(os.path.join(save_folder, f"tokenized_for_detector"))
        if "BGL" in key or "Thunderbird" in key or "openstack" in key:
            os.link(os.path.join(save_folder, f"tokenized/{key}.pkl"), os.path.join(save_folder, f"tokenized_for_detector/{key}.pkl")) 


def tokenize_HDFS_for_detector(data_path, tokenizer, save_path, max_log_lines):
    print("Tokenizing HDFS with the block structure for the detector.")
    good_blocks = []
    block_names = []
    labels = []

    with open(os.path.join(data_path, "HDFS_v1/preprocessed/anomaly_label.csv")) as file:
        for line in file:
            if line.split(",")[-1].strip() == "Normal":
                good_blocks.append(line.split(",")[0].strip())
            block_names.append(line.split(",")[0].strip())
            labels.append(line.strip())

    if not os.path.exists(os.path.join(save_path, "labels")):
        os.makedirs(os.path.join(save_path, "labels"))
    with open(os.path.join(save_path, "labels/anomaly_label_HDFS_1.pkl"), "wb") as save_file:
        pickle.dump(labels[1:], save_file)
    blocks = {}

    for b in block_names:
        blocks[b] = []

    with open(os.path.join(data_path, "HDFS_v1/HDFS.log")) as file:
        idx = 0
        for line in file:
            if idx == max_log_lines:
                break
            idx += 1
            if idx % 25000 == 0:
                print(f"At index={idx}", end="\r")
            for word in line.split(" "):
                if word.startswith("blk_"):
                    blocks[word.strip().replace(".", "")].append(line)
    
    less_than_20 = []
    encoded_dict = {}

    for key in blocks.keys():
        encoded = []
        for line in blocks[key]:
            line = line.strip()
            line = "[CLS]" + line + "[SEP]"
            encoded.append(tokenizer.encode(line).ids)
        if len(blocks[key]) > 20:
            encoded_dict[key] = encoded
        else:
            less_than_20.append(key)
    
    print(f"Throwing away {len(less_than_20)} blocks (length <= 20).")
    if not os.path.exists(os.path.join(save_path, f"tokenized_for_detector")):
        os.makedirs(os.path.join(save_path, f"tokenized_for_detector"))
    with open(os.path.join(save_path, "tokenized_for_detector/HDFS_1_block_dict.pkl"), "wb") as save_file:
        pickle.dump(encoded_dict, save_file)


def tokenize_Hadoop_for_detector(data_path, tokenizer, save_path, max_log_lines):
    print("Tokenizing Hadoop with the block structure for the detector.")
    labels = []
    in_normal = False
    with open(os.path.join(data_path, "Hadoop/abnormal_label.txt")) as file:
        for line in file:
            if line.strip() == "Normal:":
                in_normal = True
            elif in_normal:
                if "application_" in line:
                    labels.append(f"{line.strip().split(' ')[1]},Normal")
                else:
                    in_normal = False
            elif not in_normal and "application_" in line:
                labels.append(f"{line.strip().split(' ')[1]},Abnormal")
    print(f"Found {len(labels)} number of applications in Hadoop.")

    if not os.path.exists(os.path.join(save_path, "labels")):
        os.makedirs(os.path.join(save_path, "labels"))
    with open(os.path.join(save_path, "labels/abnormal_label_hadoop.pkl"), "wb") as save_file:
        pickle.dump(labels, save_file)

    hadoop_apps = {}

    number_of_lines = 0
    hadoop_path = os.path.join(data_path, "Hadoop")
    for file in os.listdir(hadoop_path):
        if "application" in file:
            for innerfile in os.listdir(os.path.join(hadoop_path,file)):
                hadoop_apps[file + "_" + innerfile.split(".")[0]] = []
                with open(os.path.join(os.path.join(hadoop_path,file),innerfile)) as logfile:
                    for line in logfile:
                        hadoop_apps[file + "_" + innerfile.split(".")[0]].append(line.strip())
                        number_of_lines += 1
        if number_of_lines >= max_log_lines:
            break

    encoded_dict = {}
    less_than_20 = 0

    for key in hadoop_apps.keys():
        encoded = []
        for line in hadoop_apps[key]:
            line = line.strip()
            line = "[CLS]" + line + "[SEP]"
            encoded.append(tokenizer.encode(line).ids)
        if len(encoded) > 20:
            encoded_dict[key] = encoded
        else:
            less_than_20 += 1
    print(F"Threw away {less_than_20} blocks with messages <= 20.")
    if not os.path.exists(os.path.join(save_path, f"tokenized_for_detector")):
        os.makedirs(os.path.join(save_path, f"tokenized_for_detector"))
    with open(os.path.join(save_path, "tokenized_for_detector/Hadoop_for_detector_block_dict.pkl"), "wb") as save_file:
        pickle.dump(encoded_dict, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='data_preprocess',
        description='This script preprocesses the datasets downloaded from Loghub. \n ascii_policy: remove/special_char \n num_policy: special_char/keep',
    )
    parser.add_argument('-d', '--data_folder', required=True)
    parser.add_argument('-s', '--save_folder', required=True) 
    parser.add_argument('-v', '--vocab_size', default=1000, type=int) 
    parser.add_argument('-l', '--max_log_lines', default=5000000, type=int)
    parser.add_argument('-a', '--ascii_policy', required=True, choices=["remove", "special_char"])
    parser.add_argument('-n', '--num_policy', required=True, choices=["0_9_special_char", "num_special_char"])
    args = parser.parse_args()

    # Create the save folder if it doesn't exist
    if not os.path.exists(args.save_folder):
        print(f"Creating {args.save_folder} folder as it doesn't exist...")
        os.makedirs(args.save_folder)

    data_to_tokenize = get_data_to_tokenize(data_folder=args.data_folder, max_log_lines=args.max_log_lines, save_path=args.save_folder)

    print("data_to_tokenize", len(data_to_tokenize["HDFS"]))

    # Create dataframe and suffle rows to mix data for the tokeniaztion
    data_df = pd.DataFrame({'log_text': [item for sublist in data_to_tokenize.values() for item in sublist]})

    if args.ascii_policy == "remove":
        print("Removing non-ascii characters from logs...")
        data_df['log_text'] = data_df['log_text'].apply(lambda x : remove_non_ascii(x))
    elif args.ascii_policy == "special_char":
        print("Replacing non-ascii characters with special token [NONA]...")
        data_df['log_text'] = data_df['log_text'].apply(lambda x : replace_non_ascii(x))
    else:
        raise Exception("Wrong ascii policy set in arg")

    if args.num_policy == "num_special_char":
        print("Replacing numbers with the [NUM] special character in the logs...")
        data_df['log_text'] = data_df['log_text'].apply(lambda x : replace_num(x))
    elif args.num_policy == "0_9_special_char":
        print("Keeping numbers as special characters in logs...")
    else:
        raise Exception("Wrong num policy set in arg")

    trained_tokenizer = train_tokenizer(vocab_size=args.vocab_size, save_folder=args.save_folder, data_df=data_df, ascii_policy=args.ascii_policy, num_policy=args.num_policy)

    tokenize_files(files_dict=data_to_tokenize, save_folder=args.save_folder, trained_tokenizer=trained_tokenizer)

    tokenize_HDFS_for_detector(data_path=args.data_folder, tokenizer=trained_tokenizer, save_path=args.save_folder, max_log_lines=args.max_log_lines)

    tokenize_Hadoop_for_detector(data_path=args.data_folder, tokenizer=trained_tokenizer, save_path=args.save_folder, max_log_lines=args.max_log_lines)
