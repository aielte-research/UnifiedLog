import torch
from torch import randperm
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import StepLR


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def forward(self, x, max_len):
        x += self.pos_encoding[:, :max_len, :].to("cuda")
        return x
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return torch.tensor(pos_encoding, dtype=torch.float32)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, max_len, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.att = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim, batch_first=True) # (batch_size, max_len, embed_dim)

        self.sequential = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=ff_dim, out_features=embed_dim)
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate, inplace=False)
        self.dropout2 = nn.Dropout(rate, inplace=False)

    def forward(self, t_inputs):
        attn_output, _ = self.att(t_inputs, t_inputs, t_inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(t_inputs + attn_output)
        seq_output = self.sequential(out1)
        seq_output = self.dropout2(seq_output)
        return self.layernorm2(out1 + seq_output)


class NeuralLog(nn.Module):
    def __init__(self, embed_dim, ff_dim, max_len, num_heads, dropout):
        super(NeuralLog, self).__init__()
        self.max_len = max_len
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, max_len)
        self.embedding_layer = PositionEmbedding(max_len, embed_dim)
        self.avgpool1d = nn.AvgPool2d((1,1), stride=1)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features=embed_dim, out_features=32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, inputs):
        x = self.embedding_layer(inputs, self.max_len)
        x = self.transformer_block(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class EarlyStopping():
    def __init__(self, tolerance, min_delta):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = model
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            # Save best model
            self.best_model = model
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.tolerance}")
            if self.counter >= self.tolerance:
                print('INFO: Early stopping')
                self.early_stop = True


class BatchGenerator(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx): 
        return self.X[idx], self.Y[idx]
    

class AnomalyDetector():
    def __init__(self, load_path, run, embed_dim, ff_dim, max_len, num_heads, dropout) -> None:
        if load_path is None:
            self.model = NeuralLog(
                embed_dim=embed_dim, 
                ff_dim=ff_dim,
                max_len=max_len, 
                num_heads=num_heads, 
                dropout=dropout
            )
        else:
            print(f"Loading anomaly detector...(path={load_path})")
            self.model = pickle.load(open(load_path, "rb"))
        self.max_len = max_len
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.run = run
    
    def balancing(self, x, y, ratio=1):      
        print(f"All sampes: {len(y)} anomalious samples: {sum(y)}")
        print("Balancing the positive negative class ratio...")

        pos_idx = [i for i, l in enumerate(y) if l == 1]
        neg_idx = [i for i, l in enumerate(y) if l == 0]
        # Thorw exception if only one type of label exists
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise Exception("Only one label is found in the training data of the detector.")
        if len(pos_idx) < len(neg_idx):
            pos_idx = torch.tensor(pos_idx)
            neg_idx = torch.tensor(neg_idx)
            pos_idx = pos_idx[randperm(len(pos_idx))]
            neg_idx = neg_idx[randperm(len(neg_idx))]
            
            neg_idx = neg_idx[:int(len(pos_idx)*ratio)]

            check_ids = [False] * len(x)
            for idx in pos_idx:
                check_ids[idx] = True

            for idx in neg_idx:
                check_ids[idx] = True

            x = [s for i, s in enumerate(x) if check_ids[i]]
            y = [s for i, s in enumerate(y) if check_ids[i]]
            return x, y
        else:
            pos_idx = torch.tensor(pos_idx)
            neg_idx = torch.tensor(neg_idx)
            pos_idx = pos_idx[randperm(len(pos_idx))]
            neg_idx = neg_idx[randperm(len(neg_idx))]
            
            pos_idx = pos_idx[:int(len(neg_idx)*ratio)]

            check_ids = [False] * len(x)
            for idx in pos_idx:
                check_ids[idx] = True

            for idx in neg_idx:
                check_ids[idx] = True

            x = [s for i, s in enumerate(x) if check_ids[i]]
            y = [s for i, s in enumerate(y) if check_ids[i]]
            return x, y


    def train(self, X, Y, train_val_test_split, save_path, lr, lr_decay_step_size, lr_decay_gamma, early_stop_tolerance, early_stop_min_delta, batch_size, epochs, balancing_ratio):
        t0 = time.time()
        _X = {}
        _Y = {}
        for key in X.keys():
            _X[key] = []
            _Y[key] = []
        
        for key in X.keys():
            if "HDFS_1" in key or "hadoop" in key.lower():
                if "hadoop" in key:
                    hadoop_normal_indexes = []
                for block in X[key].keys():
                    if Y[key][block] == 0:
                        normal_start_ind = len(_X[key])+1
                    for i in range(len(X[key][block])-self.max_len):
                        x = X[key][block][i : min(i + self.max_len, len(X[key][block]))]
                        y = torch.tensor(Y[key][block])
                        _X[key].append(x)
                        _Y[key].append(y)
                    if Y[key][block] == 0 and "hadoop" in key:
                        hadoop_normal_indexes.append([normal_start_ind,len(_X[key])])
            else:
                for i in range(len(X[key])-self.max_len):
                    x = X[key][i : min(i + self.max_len, len(X[key]))]
                    y = torch.max(Y[key][i : min(i + self.max_len, len(Y[key]))])
                    _X[key].append(x)
                    _Y[key].append(y)

        _Y_train = []
        _X_train = []
        _Y_val = {}
        _X_val = {}
        self._Y_test = {}
        self._X_test = {}
        for key in X.keys():
            _Y_val[key] = []
            _X_val[key] = []
            self._Y_test[key] = []
            self._X_test[key] = []
            if "hadoop" in key:
                for s_index, e_index in hadoop_normal_indexes[:int(len(hadoop_normal_indexes) * train_val_test_split[0])]:
                    _Y_train += _Y[key][s_index:e_index]
                    _X_train += _X[key][s_index:e_index]
                
                for s_index, e_index in hadoop_normal_indexes[int(len(hadoop_normal_indexes) * train_val_test_split[0]):int(len(hadoop_normal_indexes) * train_val_test_split[1])]:
                    _Y_val[key] += _Y[key][s_index:e_index]
                    _X_val[key] += _X[key][s_index:e_index]

                self._Y_test[key] = []
                self._X_test[key] = []
                for s_index, e_index in hadoop_normal_indexes[int(len(hadoop_normal_indexes) * train_val_test_split[1]):]:
                    self._Y_test[key] += _Y[key][s_index:e_index]
                    self._X_test[key] += _X[key][s_index:e_index]
                    
                for s_index, e_index in reversed(hadoop_normal_indexes): 
                    for idx in range(s_index,e_index, 1):
                        _Y[key].pop(idx)
                        _X[key].pop(idx)
                
            _Y_train += _Y[key][:int(len(_Y[key]) * train_val_test_split[0])]
            _X_train += _X[key][:int(len(_X[key]) * train_val_test_split[0])]
            _Y_val[key] += _Y[key][int(len(_Y[key]) * train_val_test_split[0]):int(len(_Y[key]) * train_val_test_split[1])]
            _X_val[key] += _X[key][int(len(_X[key]) * train_val_test_split[0]):int(len(_X[key]) * train_val_test_split[1])]
            self._Y_test[key] += _Y[key][int(len(_Y[key]) * train_val_test_split[1]):]
            self._X_test[key] += _X[key][int(len(_X[key]) * train_val_test_split[1]):]

        ### Manage openstack
        for key in X.keys():
            if "openstack" in key:
                _Y_val["openstack"] = []
                _X_val["openstack"] = []
                self._Y_test["openstack"] = []
                self._X_test["openstack"] = []
                break
        
        for key in X.keys():
            if "openstack" in key:
                _Y_val["openstack"] += _Y_val[key]
                _X_val["openstack"] += _X_val[key]
                self._Y_test["openstack"] += self._Y_test[key]
                self._X_test["openstack"] += self._X_test[key]
                _Y_val.pop(key)
                _X_val.pop(key)
                self._Y_test.pop(key)
                self._X_test.pop(key)

        _X_train, _Y_train = self.balancing(_X_train, _Y_train, ratio=balancing_ratio)
        _Y_train = torch.stack(_Y_train)
        _X_train = torch.stack(_X_train)

        print(f"Train samples:\t\t {len(_Y_train)}")
        print(f"Validation samples:\t {sum([len(_Y_val[key]) for key in _Y_val.keys()])}")
        print(f"Test samples:\t\t {sum([len(self._Y_test[key]) for key in self._Y_test.keys()])}")

        if self.run is not None:
            self.run["anomaly_detector"]["train_samples"] = len(_Y_train)
            for key in _X_val.keys():
                self.run["anomaly_detector"]["val_samples"][key] = len(_Y_val[key])
                self.run["anomaly_detector"]["test_samples"][key] = len(self._Y_test[key])

        train_generator = BatchGenerator(
            _X_train,
            _Y_train,
        )
        train_loader = DataLoader(train_generator, num_workers=0, batch_size=batch_size, shuffle=True)
        val_loaders = {}
        for key in _X_val.keys():
            val_generator = BatchGenerator(
                _X_val[key],
                _Y_val[key],
            )
            val_loader = DataLoader(val_generator, num_workers=0, batch_size=batch_size, shuffle=True)
            val_loaders[key] = val_loader

        self.model = self.model.to("cuda")

        loss_object = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        if lr_decay_step_size is not None:
            scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_gamma)

        early_stopping = EarlyStopping(tolerance=early_stop_tolerance, min_delta=early_stop_min_delta)     
        
        losses = []
        val_losses = {}
        val_precisions = {}
        val_recalls = {}
        val_f1s = {}

        running_loss = 0.0
        running_val_loss = {}
        for key in _X_val.keys():
            running_val_loss[key] = 0.0
            val_losses[key] = []
            val_precisions[key] = []
            val_recalls[key] = []
            val_f1s[key] = []

        step = 0
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                step += 1
                self.model = self.model.train()
                inputs = inputs.float().to("cuda")
                targets = targets.float().to("cuda")
                optimizer.zero_grad()
                yhat = self.model(inputs)
                loss = loss_object(yhat, targets.long())
                running_loss += loss.item() * inputs.size(0)
                loss.backward()
                optimizer.step()

                if (step*batch_size) % min([1000000,len(train_loader)*batch_size]) < batch_size: 
                    self.model = self.model.eval()
                    with torch.no_grad():
                        accuracy = {}
                        report = {}
                        for key in val_loaders.keys():
                            val_preds = []
                            val_truth = []
                            correct = 0
                            total = 0
                            
                            for j, (val_inputs, val_targets) in enumerate(val_loaders[key]):
                                for k in val_targets:
                                    val_truth.append(k.float())
                                val_inputs = val_inputs.float().to("cuda")
                                val_targets = val_targets.float().to("cuda")
                                val_yhat = self.model(val_inputs)
                                _, predicted = torch.max(val_yhat.data, 1)
                                running_val_loss[key] += loss_object(val_yhat, val_targets.long()).item() * val_inputs.size(0)

                                for k in predicted:
                                    val_preds.append(k.cpu())

                                total += len(val_targets)
                                correct += (predicted == val_targets).sum()


                            accuracy[key] = 100 * correct / total
                            val_preds = np.array(val_preds)
                            val_truth = np.array(val_truth)
                            report[key] = precision_recall_fscore_support(val_truth, val_preds, zero_division=0)
                            running_val_loss[key] = (running_val_loss[key]/len(val_loaders[key]))/batch_size

                    for key in _X_val.keys():
                        val_losses[key].append(running_val_loss[key])
                        try:
                            val_precisions[key].append(report[key][0][1])
                            val_recalls[key].append(report[key][1][1])
                            val_f1s[key].append(report[key][2][1])
                        except:
                            val_precisions[key].append(-1)
                            val_recalls[key].append(-1)

                    for key in _X_val.keys():
                        if self.run is not None:
                            self.run["anomaly_detector"]["val_loss"][key].log(running_val_loss[key])
                        try:
                            if self.run is not None:
                                self.run["anomaly_detector"]["val_precision"][key].log(report[key][0][1])
                                self.run["anomaly_detector"]["val_recall"][key].log(report[key][1][1])
                                self.run["anomaly_detector"]["val_f1"][key].log(report[key][2][1])
                            val_precisions[key].append(report[key][0][1])
                            val_recalls[key].append(report[key][1][1])
                            val_f1s[key].append(report[key][2][1])
                        except:
                            if self.run is not None:
                                self.run["anomaly_detector"]["val_precision"][key].log(0)
                                self.run["anomaly_detector"]["val_recall"][key].log(0)
                                self.run["anomaly_detector"]["val_f1"][key].log(0)
                            val_precisions[key].append(-1)
                            val_recalls[key].append(-1)
                            val_f1s[key].append(-1)

                    average_val_loss = sum([running_val_loss[key] for key in running_val_loss.keys()])/len(running_val_loss.keys())
                    early_stopping(average_val_loss, model=self.model)
                    print(f"Epoch {epoch} step {step}\n\tAverage val loss: {average_val_loss}")
                    for key in _X_val.keys():
                        try: # if dataset has block structure
                            print(f"\tValidation {key}:\tLoss:{running_val_loss[key]:.4f}\t Accuracy:{accuracy[key]:.2f}\t Precision:{report[key][0][1]:.2f}\t Recall:{report[key][1][1]:.2f}\t F1:{report[key][2][1]:.2f}")
                        except:
                            print(f"\tValidation {key}:\tLoss:{running_val_loss[key]:.4f}\t Accuracy:{accuracy[key]:.2f}\t Precision:{-1}\t Recall:{-1}\t F1:{-1}")

                    if self.run is not None:
                        self.run["anomaly_detector"]["average_val_loss"].log(average_val_loss)
                    if early_stopping.early_stop:
                        if self.run is not None:
                            self.run["anomaly_detector"]["early_stop"] = f"epoch={epoch}"
                            self.model = early_stopping.best_model
                        return
                
                    running_loss = running_loss/len(train_loader)
                    print(f"Epoch {epoch}\t train loss:  {running_loss:.5f}")
                    losses.append(running_loss)
                    if self.run is not None:
                        self.run["anomaly_detector"]["train_loss"].log(running_loss)
                    

                    running_loss = 0
                    running_val_loss = {}
                    for key in _X_val.keys():
                        running_val_loss[key] = 0.0

                    if save_path is not None:
                        if "." in save_path:
                            save_path = ".".join(save_path.split(".")[:-1])
                        model_info = "_heads_" + str(self.num_heads) + "_ff_dim_" + str(self.ff_dim) + "_maxlen_" + str(self.max_len) + "_embed_dim_" + str(self.embed_dim) + ".pkl"
                        pickle.dump(self.model, open(save_path + model_info, "wb"))
                        if self.run is not None:
                            self.run["anomaly_detector"]["save_path"] = save_path + model_info

                    if early_stopping.early_stop and save_path is not None:
                        print(f"Early stopping...")
                        if self.run is not None:
                            self.run["anomaly_detector"]["early_stop"] = f"epoch={epoch}"
                        break
                    if lr_decay_step_size is not None:
                        scheduler.step() # learning rate decay  


        print(f"Training took: {time.time()-t0}s")

    def eval(self, batch_size, X_test=None, Y_test=None):
        print("Evaluating anomaly detector...")
        _X = {}
        _Y = {}
        for key in X_test.keys():
            _X[key] = []
            _Y[key] = []
        
        for key in X_test.keys():
            if "HDFS_1" in key or "hadoop" in key.lower():
                if "hadoop" in key:
                    hadoop_normal_indexes = []
                for block in X_test[key].keys():
                    if Y_test[key][block] == 0:
                        normal_start_ind = len(_X[key])+1
                    for i in range(len(X_test[key][block])-self.max_len):
                        x = X_test[key][block][i : min(i + self.max_len, len(X_test[key][block]))]
                        y = torch.tensor(Y_test[key][block])
                        _X[key].append(x)
                        _Y[key].append(y)
                    if Y_test[key][block] == 0 and "hadoop" in key:
                        hadoop_normal_indexes.append([normal_start_ind,len(_X[key])])
            else:
                for i in range(len(X_test[key])-self.max_len):
                    x = X_test[key][i : min(i + self.max_len, len(X_test[key]))]
                    y = torch.max(Y_test[key][i : min(i + self.max_len, len(Y_test[key]))])
                    _X[key].append(x)
                    _Y[key].append(y)

        for key in X_test.keys():
            self._Y_test[key] = []
            self._X_test[key] = []
            if "hadoop" in key:

                self._Y_test[key] = []
                self._X_test[key] = []
                for s_index, e_index in hadoop_normal_indexes:
                    self._Y_test[key] += _Y[key][s_index:e_index]
                    self._X_test[key] += _X[key][s_index:e_index]
                    
                for s_index, e_index in reversed(hadoop_normal_indexes): 
                    for idx in range(s_index,e_index, 1):
                        _Y[key].pop(idx)
                        _X[key].pop(idx)
                
            self._Y_test[key] += _Y[key]
            self._X_test[key] += _X[key]

        ### Manage openstack
        for key in X_test.keys():
            if "openstack" in key:
                self._Y_test["openstack"] = []
                self._X_test["openstack"] = []
                break
        
        for key in X_test.keys():
            if "openstack" in key:
                self._Y_test["openstack"] += _Y[key]
                self._X_test["openstack"] += _X[key]
                self._Y_test.pop(key)
                self._X_test.pop(key)
        
        with torch.no_grad():
            for key in self._X_test.keys():
                test_generator = BatchGenerator(
                    self._X_test[key],
                    self._Y_test[key],
                )
                test_loader = DataLoader(test_generator, num_workers=0, batch_size=batch_size, shuffle=False)

                preds = []
                truth = []
                self.model = self.model.eval()
                for i, (x_test, y_test) in enumerate(test_loader):
                    for y in y_test:
                        truth.append(y.item())
                    y_hat = self.model(x_test.float().cuda())
                    for y_h in y_hat:
                        preds.append(torch.argmax(y_h,-1).cpu().item())
                preds = np.array(preds)
                truth = np.array(truth)
                report = precision_recall_fscore_support(truth, preds, zero_division=0)
                try:
                    print(f"Testing model on {key}: precision: {report[0][1]} recall: {report[1][1]} f1: {report[2][1]} | anomaly/all {np.sum(truth == 1)}/{len(truth)}")
                    if self.run is not None:
                        self.run["anomaly_detector"]["test"][key]["precision"]   = report[0][1]
                        self.run["anomaly_detector"]["test"][key]["recall"]      = report[1][1]
                        self.run["anomaly_detector"]["test"][key]["f1"]          = report[2][1]
                except:
                    print(f"No anomalies in test data, out of {len(truth)} elements")
                