import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics


def build(train_dataset,test_dataset,root,num_class):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if test_dataset == "Kvasir":
        img_path = root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    if test_dataset == "CVC":
        img_path = root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    if test_dataset == "ETIS-Larib":
        img_path = root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    if test_dataset == "CVC-ColonDB":
        img_path = root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    if test_dataset == "CVC-300":
        img_path = root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    if test_dataset == "2018DSB":
        img_path = root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    if test_dataset == "ISIC2018":
        img_path = root + "ISIC2018_Task1-2_Training_Input/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = root + "ISIC2018_Task1_Training_GroundTruth/*"
        target_paths = sorted(glob.glob(depth_path))
    _, test_dataloader, _ = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=1, num_class=num_class,
    )

    _, test_indices, val_indices = dataloaders.split_ids(len(target_paths))
    target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

    perf = performance_metrics.DiceScore()

    model = models.Net(num_class, 352)

    state_dict = torch.load(
        "./Trained models/net_30Kvasir90.pt".format(train_dataset)
    )
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, test_dataloader, perf, model, target_paths


@torch.no_grad()
def predict(train_dataset,test_dataset,root,num_class):
    device, test_dataloader, perf_measure, model, target_paths = build(train_dataset,test_dataset,root,num_class)

    if not os.path.exists("./Predictions"):
        os.makedirs("./Predictions")
    if not os.path.exists("./Predictions/Trained on {}".format(train_dataset)):
        os.makedirs("./Predictions/Trained on {}".format(train_dataset))
    if not os.path.exists(
        "./Predictions/Trained on {}/Tested on {}".format(
            train_dataset, test_dataset
        )
    ):
        os.makedirs(
            "./Predictions/Trained on {}/Tested on {}".format(
                train_dataset, test_dataset
            )
        )

    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        # output = model(data)
        output,_,_,_ = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        if num_class == 1:
            predicted_map = predicted_map > 0
        else:
            rgb_predicted_map = np.zeros(predicted_map.shape)
            bgr_predicted_map = []

            for i in range(predicted_map.shape[1]):
                for j in range(predicted_map.shape[2]):
                    rgb_predicted_map[np.argmax(predicted_map[:, i, j], axis=0), i, j] = 1

            rgb_predicted_map = np.transpose(rgb_predicted_map, (1, 2, 0))
            bgr_predicted_map = rgb_predicted_map[: , :, [2, 1, 0]]
            predicted_map = bgr_predicted_map

        cv2.imwrite(
            "./Predictions/Trained on {}/Tested on {}/{}".format(
                train_dataset, test_dataset, os.path.basename(target_paths[i])
            ),
            predicted_map * 255,
        )
        if i + 1 < len(test_dataloader):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )


# def get_args():
#     parser = argparse.ArgumentParser(
#         description="Make predictions on specified dataset"
#     )
#     parser.add_argument(
#         "--train-dataset", type=str, required=True, choices=["Kvasir", "CVC"]
#     )
#     parser.add_argument(
#         "--test-dataset", type=str, required=True, choices=["Kvasir", "CVC"]
#     )
#     parser.add_argument("--data-root", type=str, required=True, dest="root")
#
#     return parser.parse_args()


def main():
    # args = get_args()
    ###参数###
    num_class = 1
    train_dataset = "Kvasir"
    test_dataset = "Kvasir"
    root = "./polyp_original/Kvasir-SEG/"
    ###参数###
    predict(train_dataset,test_dataset,root,num_class)


if __name__ == "__main__":
    main()
