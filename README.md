# EDO-SANet

### Preparation
- Install the requirements:

```
pip install -r requirements.txt
```

- Compiling CUDA operators
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

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
```

+ Replace `[train data]` with training dataset name (options: `Kvasir`; `CVC`; `Kvasir+CVC`; `2018DSB`; `ISIC2018`).

+ Replace `[test data]` with testing dataset name (options: `Kvasir`; `CVC`; `2018DSB`; `ISIC2018`;`CVC-ColonDB`; `ETIS-Larib`; `EndoScene-CVC300`).


### Evaluation

Evaluate pre-computed predictions from a trained model for a test split. Note, the test split can be from a different dataset to the train split:

```
python eval.py --train-dataset=[train data] --test-dataset=[test data] --data-root=[path]
```

+ Replace `[train data]` with training dataset name (options: `Kvasir`; `CVC`; `Kvasir+CVC`; `2018DSB`; `ISIC2018`).

+ Replace `[test data]` with testing dataset name (options: `Kvasir`; `CVC`; `2018DSB`; `ISIC2018`;`CVC-ColonDB`; `ETIS-Larib`; `EndoScene-CVC300`).

###  Acknowledgement
We are very grateful for these excellent works[FCBFormer](https://github.com/ESandML/FCBFormer) and [InternImage](https://github.com/OpenGVLab/InternImage), which have provided the basis for our framework.
