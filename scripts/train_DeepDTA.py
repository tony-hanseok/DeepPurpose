import os

import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

X_drug, X_target, y = load_process_KIBA("./data/", binary=False)

drug_encoding = "CNN"
target_encoding = "CNN"
train, val, test = data_process(
    X_drug,
    X_target,
    y,
    drug_encoding,
    target_encoding,
    split_method="random",
    frac=[0.7, 0.1, 0.2],
)

# use the parameters setting provided in the paper: https://arxiv.org/abs/1801.10193
config = generate_config(
    drug_encoding=drug_encoding,
    target_encoding=target_encoding,
    cls_hidden_dims=[1024, 1024, 512],
    train_epoch=100,
    LR=0.001,
    batch_size=256,
    cnn_drug_filters=[32, 64, 96],
    cnn_target_filters=[32, 64, 96],
    cnn_drug_kernels=[4, 6, 8],
    cnn_target_kernels=[4, 8, 12],
)

model = models.model_initialize(**config)
model.train(train, val, test)
model.save_model(model.result_folder)