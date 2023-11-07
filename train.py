import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import loss


def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss, wiou_loss, wbce_loss, num_class):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # output = model(data)
        output,output1,output2,output3 = model(data)

        if num_class==1:
            loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
            # loss = wiou_loss(output, target) + wbce_loss(output, target)
            #deep supervision
            loss1 = Dice_loss(output1, target) + BCE_loss(torch.sigmoid(output1), target)
            loss2 = Dice_loss(output2, target) + BCE_loss(torch.sigmoid(output2), target)
            loss3 = Dice_loss(output3, target) + BCE_loss(torch.sigmoid(output3), target)
            # loss1 = wiou_loss(output1, target) + wbce_loss(output1, target)
            # loss2 = wiou_loss(output2, target) + wbce_loss(output2, target)
            # loss3 = wiou_loss(output3, target) + wbce_loss(output3, target)
            
            loss = 0.6*loss+0.2*loss1+0.1*loss2+0.1*loss3
     
        else:
            loss = Dice_loss(output, target) + BCE_loss(output, target)

        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def atest(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # output = model(data)
        #depp supervision
        output,_,_,_ = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(perf_accumulator), np.std(perf_accumulator)

###参数###
dataset="Kvasir"
#choices=["Kvasir", "CVC"]
data_root="./polyp_original/Kvasir-SEG/"
num_class=1
epochs=200
batchsize=16
lr=1e-4
lrs=True
lrs_min=1e-6
multi_gpu=False
###参数###

def build(dataset,data_root,batchsize,lr,multi_gpu,num_class):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if dataset == "Kvasir":
        img_path = data_root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = data_root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    if dataset == "CVC":
        img_path = data_root+ "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = data_root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))

    train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=batchsize, num_class=num_class,
    )

    Dice_loss = loss.SoftDiceLoss()
    wiou_loss = loss.wIOU_loss()
    wbce_loss = loss.wBCE_loss()
    if num_class==1:
        BCE_loss = nn.BCELoss()
    else:
        BCE_loss = loss.CrossEntropyLoss()

    perf = performance_metrics.DiceScore()

    model = models.Net(num_class, 352)

    if multi_gpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    return (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        wiou_loss,
        wbce_loss,
        perf,
        model,
        optimizer,
    )


def train(dataset,data_root,batchsize,lr,multi_gpu,num_class):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        wiou_loss,
        wbce_loss,
        perf,
        model,
        optimizer,
    ) = build(dataset,data_root,batchsize,lr,multi_gpu,num_class)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = None
    if lrs == "true":
        if lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss, wiou_loss, wbce_loss, num_class
            )
            test_measure_mean, test_measure_std = atest(
                model, device, val_dataloader, epoch, perf
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if multi_gpu == "false"
                    else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                "Trained models/net_30" + dataset + str(epoch) + ".pt",
            )
            prev_best_test = test_measure_mean
            
        if epoch%10==0:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if multi_gpu == "false"
                    else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                "Trained models/net_30" + dataset + str(epoch) + ".pt",
            )


# def get_args():
#     parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
#     parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir", "CVC"])
#     parser.add_argument("--data-root", type=str, required=True, dest="root")
#     parser.add_argument("--epochs", type=int, default=200)
#     parser.add_argument("--batch-size", type=int, default=16)
#     parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
#     parser.add_argument(
#         "--learning-rate-scheduler", type=str, default="true", dest="lrs"
#     )
#     parser.add_argument(
#         "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
#     )
#     parser.add_argument(
#         "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
#     )
#
#     return parser.parse_args()


def main():
    # args = get_args()
    train(dataset,data_root,batchsize,lr,multi_gpu,num_class)


if __name__ == "__main__":
    main()
