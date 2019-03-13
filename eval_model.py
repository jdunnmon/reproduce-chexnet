import torch
import pandas as pd
import cxr_dataset as CXR
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np


def make_pred_multilabel(data_input, model, PATH_TO_IMAGES, loader=False, save=False):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_input: torchvision transforms to preprocess raw images; same as validation transforms, 
                    or dataloader
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which NIH images can be found
        loader: if True, data_input handled as datalaoder; otherwise, as tranform list
        save: save results to csv
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 16

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create dataloader if needed (default behavior)
    if not loader:
        dataset = CXR.CXRDataset(
            path_to_images=PATH_TO_IMAGES,
            fold="test",
            transform=data_input['val'])
        dataloader = torch.utils.data.DataLoader(
            dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    else:
        dataloader = data_input
        dataset = dataloader.dataset
        dataloader.batch_sampler.batch_size = BATCH_SIZE
    
    size = len(dataset)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()
        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            if (BATCH_SIZE * i + j) < len(dataset.df):
                thisrow = {}
                truerow = {}
                thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
                truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

                # iterate over each entry in prediction vector; each corresponds to
                # individual label
                for k in range(len(dataset.PRED_LABEL)):
                    thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                    truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

                pred_df = pred_df.append(thisrow, ignore_index=True)
                true_df = true_df.append(truerow, ignore_index=True)
            else:
                print('Over dataset size, ignoring...')
                print(BATCH_SIZE*i+j)

        if not (i % 100):
            print('Evaluated '+str(i * BATCH_SIZE)+' of '+str(len(dataset))+' Examples...')

    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calc AUCs
    for column in true_df:

        if column not in [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
                'Hernia']:
                    continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        thisrow['auc'] = np.nan
        try:
            thisrow['auc'] = sklm.roc_auc_score(
                actual.as_matrix().astype(int), pred.as_matrix())
        except BaseException:
            print("can't calculate auc for " + str(column))
        auc_df = auc_df.append(thisrow, ignore_index=True)
    if save:
        pred_df.to_csv("results/preds.csv", index=False)
        auc_df.to_csv("results/aucs.csv", index=False)
    return pred_df, auc_df
