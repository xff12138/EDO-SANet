# EDO-SANet

### Training

Train EDO-SANet on the train split of a dataset:

```
python train.py --dataset=[train data] --data-root=[path]
```

+ Replace `[train data]` with training dataset name (options: `Kvasir`; `CVC`; `Kvasir+CVC`; `2018DSB`; `ISIC2018`).


### Prediction

Generate predictions from a trained model for a test split. Note, the test split can be from a different dataset to the train split:

```
python predict.py --train-dataset=[train data] --test-dataset=[test data] --data-root=[path]

+ Replace `[train data]` with training dataset name (options: `Kvasir`; `CVC`; `Kvasir+CVC`; `2018DSB`; `ISIC2018`).

+ Replace `[test data]` with testing dataset name (options: `Kvasir`; `CVC`; `2018DSB`; `ISIC2018`;`CVC-ColonDB`; `ETIS-Larib`; `EndoScene-CVC300`).


### Evaluation

Evaluate pre-computed predictions from a trained model for a test split. Note, the test split can be from a different dataset to the train split:

```
python eval.py --train-dataset=[train data] --test-dataset=[test data] --data-root=[path]
```

+ Replace `[train data]` with training dataset name (options: `Kvasir`; `CVC`; `Kvasir+CVC`; `2018DSB`; `ISIC2018`).

+ Replace `[test data]` with testing dataset name (options: `Kvasir`; `CVC`; `2018DSB`; `ISIC2018`;`CVC-ColonDB`; `ETIS-Larib`; `EndoScene-CVC300`).

