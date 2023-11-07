import torch
import glob
from Data.dataloaders import split_ids
from skimage.io import imread
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, mean_squared_error
import numpy as np
from skimage.transform import resize
# print(torch.cuda.is_available())


train_dataset = "Kvasir"
test_dataset = "Kvasir"
root = "./polyp_original/Kvasir-SEG/"

if test_dataset == "Kvasir":
    prediction_files = sorted(
        glob.glob(
            "./Predictions/Trained on {}/Tested on {}/*".format(
                train_dataset, test_dataset
            )
        )
    )
    depth_path = root + "masks/*"
    target_paths = sorted(glob.glob(depth_path))
elif test_dataset == "CVC":
    prediction_files = sorted(
        glob.glob(
            "./Predictions/Trained on {}/Tested on {}/*".format(
                train_dataset, test_dataset
            )
        )
    )
    depth_path = root + "Ground Truth/*"
    target_paths = sorted(glob.glob(depth_path))

_, test_indices, val_indices = split_ids(len(target_paths))
test_files = sorted(
    [target_paths[test_indices[i]] for i in range(len(test_indices))]
)

dice = []
IoU = []
precision = []
recall = []
mae = []
for i in range(len(test_files)):
    pred = np.ndarray.flatten(imread(prediction_files[i]) / 255) > 0.5
    gt = (
            resize(imread(test_files[i]), (int(352), int(352)), anti_aliasing=False)
            > 0.5
    )

    if len(gt.shape) == 3:
        gt = np.mean(gt, axis=2)
    gt = np.ndarray.flatten(gt)

    dice.append(f1_score(gt, pred))
    IoU.append(jaccard_score(gt, pred))
    precision.append(precision_score(gt, pred))
    recall.append(recall_score(gt, pred))
    mae.append(mean_squared_error(gt, pred))

    if i + 1 < len(test_files):
        print(
            "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}, mae={:.6f}".format(
                i + 1,
                len(test_files),
                100.0 * (i + 1) / len(test_files),
                np.mean(dice),
                np.mean(IoU),
                np.mean(precision),
                np.mean(recall),
                np.mean(mae),
            ),
            end="",
        )
    else:
        print(
            "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}, recall={:.6f}".format(
                i + 1,
                len(test_files),
                100.0 * (i + 1) / len(test_files),
                np.mean(dice),
                np.mean(IoU),
                np.mean(precision),
                np.mean(recall),
                np.mean(mae),
            )
        )
