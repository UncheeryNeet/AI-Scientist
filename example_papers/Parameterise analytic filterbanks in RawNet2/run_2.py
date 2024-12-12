from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import random
import numpy as np
import math
import os
import sys
import soundfile as sf
import torch.utils.data as data
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Any, Optional, Tuple, List
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
from evaluation import compute_eer, calculate_tDCF_EER, produce_evaluation_file
from asteroid_filterbanks import Encoder, ParamSincFB


@dataclass
class Config:
    data_path: str = "/content/data/LA"
    num_classes: int = 10
    batch_size: int = 32 * 4
    weight_decay: float = 1e-4
    device: str = "cuda"
    out_dir: str = "run_0"
    seed: int = 1234
    track: str = "LA"
    compile_model: bool = True
    train_duration_sec: int = 4
    test_duration_sec: int = 4
    max_epoch: int = 30  # Due to bad gpu
    lr: float = 4e-4
    lr_min: float = 1e-6
    num_workers: int = 4

    wandb_disabled = False
    wandb_project: str = "ASV-Spoofing"
    wandb_name: str = "rawnet2"


@dataclass
class ModelArgs:
    nb_samp: int = 64000
    first_conv: int = 1024
    in_channels: int = 1
    nb_fc_node: int = 1024
    gru_node: int = 1024
    nb_gru_layer: int = 3
    nb_classes: int = 2
    preemp: bool = False


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        device,
        out_channels,
        kernel_size,
        in_channels=1,
        sample_rate=16000,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
    ):
        super(SincConv, self).__init__()

        if in_channels != 1:

            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        # initialize filterbanks using Mel scale
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)  # Hz to mel conversion
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel = filbandwidthsf
        self.hsupp = torch.arange(
            -(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1
        )
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate
            )
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp / self.sample_rate
            )
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            x,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            padding=1,
            kernel_size=3,
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=0,
                kernel_size=1,
                stride=1,
            )
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.95) -> None:
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert (
            len(input.size()) == 2
        ), "The number of dimensions of input tensor must be 2!"
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter)


class RawNet2(nn.Module):
    def __init__(self, arg: ModelArgs, config: Config):
        super(RawNet2, self).__init__()
        self.device = config.device
        filts = [32, [32, 32], [32, 128], [128, 128]]
        first_conv = arg.first_conv  # no. of filter coefficients
        in_channels = arg.in_channels
        blocks = [2, 4]
        nb_fc_node = arg.nb_fc_node
        gru_node = arg.gru_node
        nb_gru_layer = arg.nb_gru_layer
        nb_classes = arg.nb_classes
        preemp = True
        self.preemp = preemp

        self.preprocess = nn.Sequential(
            PreEmphasis(),
            nn.InstanceNorm1d(1, eps=1e-4, affine=True),
            Encoder(
                ParamSincFB(
                    filts[0],
                    251,
                    stride=10,
                )
            ),
        )

        self.first_bn = nn.BatchNorm1d(num_features=filts[0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts=filts[1], first=True))
        self.block1 = nn.Sequential(Residual_block(nb_filts=filts[1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts=filts[2]))
        filts[2][0] = filts[2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts=filts[2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts=filts[2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts=filts[2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(
            in_features=filts[1][-1], l_out_features=filts[1][-1]
        )
        self.fc_attention1 = self._make_attention_fc(
            in_features=filts[1][-1], l_out_features=filts[1][-1]
        )
        self.fc_attention2 = self._make_attention_fc(
            in_features=filts[2][-1], l_out_features=filts[2][-1]
        )
        self.fc_attention3 = self._make_attention_fc(
            in_features=filts[2][-1], l_out_features=filts[2][-1]
        )
        self.fc_attention4 = self._make_attention_fc(
            in_features=filts[2][-1], l_out_features=filts[2][-1]
        )
        self.fc_attention5 = self._make_attention_fc(
            in_features=filts[2][-1], l_out_features=filts[2][-1]
        )

        self.bn_before_gru = nn.BatchNorm1d(num_features=filts[2][-1])
        self.gru = nn.GRU(
            input_size=filts[2][-1],
            hidden_size=gru_node,
            num_layers=nb_gru_layer,
            batch_first=True,
        )

        self.fc1_gru = nn.Linear(in_features=gru_node, out_features=nb_fc_node)

        self.fc2_gru = nn.Linear(
            in_features=nb_fc_node, out_features=nb_classes, bias=True
        )

        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, is_test=False):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, len_seq)  # Reshape to (batch_size, sequence_length)

        x = self.preprocess(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1)  # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(
            y0.size(0), y0.size(1), -1
        )  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1)  # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(
            y1.size(0), y1.size(1), -1
        )  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1  # (batch, filter, time) x (batch, filter, 1)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1)  # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(
            y2.size(0), y2.size(1), -1
        )  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2  # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1)  # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(
            y3.size(0), y3.size(1), -1
        )  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3  # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1)  # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(
            y4.size(0), y4.size(1), -1
        )  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4  # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1)  # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(
            y5.size(0), y5.size(1), -1
        )  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5  # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        emb = self.fc1_gru(x)
        x = self.fc2_gru(emb)

        if not is_test:
            output = x
            return emb, output
        else:
            output = F.softmax(x, dim=1)
            return emb, output

    def _make_attention_fc(self, in_features, l_out_features):
        l_fc = []

        l_fc.append(nn.Linear(in_features=in_features, out_features=l_out_features))

        return nn.Sequential(*l_fc)


## end of model ###
def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()
    for line in l_meta:
        _, key, _, _, label = line.strip().split(" ")
        file_list.append(key)
        d_meta[key] = 1 if label == "bonafide" else 0
    return d_meta, file_list


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt : stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64000  # take ~4 sec audio
        self.fs = 16000

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir + f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = torch.Tensor(X_pad)
        target = self.labels[key]
        return x_inp, target


class Dataset_dev(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        self.list_IDs = list_IDs

        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64000  # take ~4 sec audio
        self.fs = 16000

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir + f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = torch.Tensor(X_pad)
        target = self.labels[key]
        return x_inp, target


class Dataset_eval(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        self.list_IDs = list_IDs

        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64000  # take ~4 sec audio
        self.fs = 16000

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir + f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = torch.Tensor(X_pad)
        target = self.labels[key]
        return x_inp, key


def evaluate_accuracy(dev_loader, model, device):
    num_total = 0.0
    num_correct = 0.0
    model.eval()
    with torch.no_grad():
        label_loader, score_loader = [], []
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            _, batch_out = model(batch_x)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            score = F.softmax(batch_out, dim=1)[:, 1]
            label_loader.append(batch_y)
            score_loader.append(score)
        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(label_loader, 0).data.cpu().numpy()
        val_eer = compute_eer(scores[labels == 1], scores[labels == 0])[0]
        val_accuracy = (num_correct / num_total) * 100
        return val_accuracy, val_eer * 100


def train_epoch(train_loader, model, optimizer, epoch, device):
    """Training"""
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()
    train_log_info = []
    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += batch_loss.item() * batch_size
        if ii % 10 == 0:
            sys.stdout.write(
                "\r epoch: {}, batch : {}, {:.2f}".format(
                    epoch, ii, (num_correct / num_total) * 100
                )
            )
            train_log_info.append(
                {
                    "iter": ii,
                    "epoch": epoch,
                    "loss": batch_loss,
                }
            )
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy, train_log_info


import time


def run(config: Config, args: ModelArgs):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir)
    # ------------------------- set model ------------------------- #
    # preprocessor = PreEmphasis(args).to(config.device)
    model = RawNet2(args, config).to(config.device)
    torch.set_float32_matmul_precision("high")
    ctx = torch.amp.autocast(device_type=config.device, dtype=torch.bfloat16)
    if config.compile_model:
        print("Compiling the model...")
        model = torch.compile(model)

    loss_fn = nn.BCELoss().to(
        config.device
    )  # DDP is not needed when a module doesn't have any parameter that requires a gradient.

    track = config.track
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = os.path.join(
        config.data_path, "ASVspoof2019_{}_train/".format(track)
    )
    dev_database_path = os.path.join(
        config.data_path, "ASVspoof2019_{}_dev/".format(track)
    )
    eval_database_path = os.path.join(
        config.data_path, "ASVspoof2019_{}_eval/".format(track)
    )

    trn_list_path = os.path.join(
        config.data_path,
        "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(track, prefix_2019),
    )
    dev_trial_path = os.path.join(
        config.data_path,
        "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(track, prefix_2019),
    )
    eval_trial_path = os.path.join(
        config.data_path,
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(track, prefix_2019),
    )

    d_label_evl, file_eval = genSpoof_list(
        dir_meta=eval_trial_path, is_train=False, is_eval=False
    )
    print("no. of eval trials", len(file_eval))
    eval_set = Dataset_eval(
        list_IDs=file_eval, labels=d_label_evl, base_dir=eval_database_path
    )
    eval_loader = DataLoader(eval_set, batch_size=config.batch_size, shuffle=False)
    del eval_set, d_label_evl
    # define train dataloader
    d_label_trn, file_train = genSpoof_list(
        dir_meta=trn_list_path, is_train=True, is_eval=False
    )
    print("no. of training trials", len(file_train))
    train_set = Dataset_train(
        list_IDs=file_train, labels=d_label_trn, base_dir=trn_database_path
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        num_workers=12,
        shuffle=True,
        drop_last=True,
    )
    del train_set, d_label_trn

    # define validation dataloader

    d_label_dev, file_dev = genSpoof_list(
        dir_meta=dev_trial_path, is_train=False, is_eval=False
    )
    print("no. of validation trials", len(file_dev))
    dev_set = Dataset_dev(
        list_IDs=file_dev, labels=d_label_dev, base_dir=dev_database_path
    )
    dev_loader = DataLoader(
        dev_set, batch_size=config.batch_size, num_workers=12, shuffle=False
    )
    del dev_set, d_label_dev
    # ------------------------- optimizer ------------------------- #
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=[0.9, 0.999],
        weight_decay=config.weight_decay,
    )
    # initialize a GradScaler. If enabled=False scaler is a no-op

    best_eer = 100.0
    og_t0 = time.time()
    t0 = time.time()
    val_log_info = []
    if not config.wandb_disabled:
        wandb.init()
    num_epochs = config.max_epoch
    print(num_epochs)
    train_log_info = []
    for epoch in range(num_epochs):
        running_loss, train_accuracy, log_info = train_epoch(
            train_loader, model, optimizer, epoch, config.device
        )
        train_log_info += log_info
        # -------------------- evaluation ----------------------- #
        if epoch % 5 == 1 or epoch == config.max_epoch:
            val_accuracy, val_err = evaluate_accuracy(dev_loader, model, config.device)
            print(f"EER: {val_err}")
            wandb.log({"EER_LA": val_err, "epoch": epoch})
            val_log_info.append({"epoch": epoch, "loss": running_loss, "eer": val_err})
            torch.save(
                model.state_dict(), os.path.join(config.out_dir, f"ckpt_{epoch}.pth")
            )
            if val_err < best_eer:
                best_eer = val_err
                wandb.log({"BestEER_LA": val_err, "epoch": epoch})
                torch.save(
                    model.state_dict(), os.path.join(config.out_dir, "best_model.pth")
                )

    print("training done")
    print(f"Best validation eer: {best_eer}")
    print(f"Total train time: {(time.time() - og_t0) / 60:.2f} mins")

    if val_err > best_eer:
        print("Using best model to perform final evaluation")
        model = RawNet2(args, config).to(config.device)
        sd = torch.load(
            os.path.join(config.out_dir, "best_model.pth"), map_location="cpu"
        )
        b = sd.copy()
        for k in sd.keys():
            b[k[10:]] = sd[k]
            del b[k]
        model.load_state_dict(b)
        if config.compile_model:
            model = torch.compile(model)

    eval_acc, _ = evaluate_accuracy(dev_loader, model, config.device)
    produce_evaluation_file(
        eval_loader,
        model,
        "cuda",
        os.path.join(config.out_dir, "result.txt"),
        eval_trial_path,
    )
    eval_eer, eval_tdcf = calculate_tDCF_EER(
        cm_scores_file=os.path.join(config.out_dir, "result.txt"),
        asv_score_file=os.path.join(
            Config.data_path,
            "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
        ),
        output_file=os.path.join(config.out_dir, "t-DCF_EER.txt"),
    )
    print(f"Test EER: {eval_eer}")
    final_info = {
        "best_val_eer": best_eer,
        "test_eer": eval_eer,
        "test_acc": eval_acc,
        "test_tdcf": eval_tdcf,
        "total_train_time": time.time() - og_t0,
    }
    return final_info, val_log_info, train_log_info


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    arg = parser.parse_args()

    datasets = [
        "ASVspoof2019"
    ]  # For now, only CIFAR-10; can add 'cifar100' in the future
    num_seeds = {"ASVspoof2019": 1}  # Change the number of seeds as desired
    config = Config()
    model_args = ModelArgs()
    config.out_dir = arg.out_dir
    all_results = {}
    final_infos = {}
    for dataset in datasets:
        final_info_list = []
        for seed_offset in range(num_seeds[dataset]):
            final_info, val_info, train_info = run(config, model_args)
            all_results[f"{dataset}_{seed_offset}_final_info"] = final_info
            all_results[f"{dataset}_{seed_offset}_train_info"] = train_info
            all_results[f"{dataset}_{seed_offset}_val_info"] = val_info
            final_info_list.append(final_info)
        final_info_dict = {
            k: [d[k] for d in final_info_list] for k in final_info_list[0].keys()
        }
        means = {f"{k}_mean": np.mean(v) for k, v in final_info_dict.items()}
        stderrs = {
            f"{k}_stderr": np.std(v) / len(v) for k, v in final_info_dict.items()
        }
        final_infos[dataset] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict,
            "batch": config.batch_size,
        }

    with open(os.path.join(arg.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(arg.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)
