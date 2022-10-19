import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
# import disturtd_utils as utils
#from dice_cofficient_loss import dice_coeff


def dice_coeff(x: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1).float()
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter
        d += (2 * inter + epsilon) / (sets_sum + epsilon)
    return d / batch_size


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes + 1
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = (torch.diag(h).sum() / h.sum()).item()
        # sensitivity
        se = ((torch.diag(h) / h.sum(1))[1]).item()
        # specificity(recall)
        sp = ((torch.diag(h) / h.sum(1))[0]).item()
        # precision
        pr = ((torch.diag(h) / h.sum(0))[1]).item()
        # F1-score
        F1 = 2 * (pr * sp) / (pr + sp)
        # # iou
        # iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        return acc_global, se, sp, F1, pr


def criterion(inputs, target, dice: bool = True):
    loss1 = 0
    if dice:
        loss1 = dice_coeff(inputs, target)
    target = target.unsqueeze(1).float()
    loss2 = nn.BCELoss()(inputs, target)
    return loss1 * 0.5 + loss2


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes + 1)
    data_loader = tqdm(data_loader)
    mask = None
    predict = None
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)
            # (B,1,H,W)
            output = model(image)

            pre_infer_dir = r"G:\My Drive\final_project\code\SA_Uet-pytorch-master\test_results\DRIVE_RESULTS\only_gan"
            if pre_infer_dir:
                if i+1 < 10:
                    f_name = f"0{i+1}_test.gif"
                else:
                    f_name = f"{i+1}_test.gif"
                pre_infer_mask = os.path.join(pre_infer_dir,f_name)
                pre_mask = Image.open(pre_infer_mask).convert('L')
                pre_mask = np.array(pre_mask)
                pre_mask[pre_mask >= 0.5] = 1
                pre_mask[pre_mask < 0.5] = 0

                output = output.cpu() + pre_mask
                output[output>1]=1
                output = output.cuda()
                truth = output.clone()
                output[output >= 0.5] = 1
                output[output < 0.5] = 0

            confmat.update(target.flatten(), output.long().flatten())
            # dice.update(output, target)
            mask = target.flatten() if mask is None else torch.cat((mask, target.flatten()))
            # 它是概率集合，不能是0，1集合
            predict = truth.flatten() if predict is None else torch.cat((predict, truth.flatten()))

    mask = mask.cpu().numpy()
    predict = predict.cpu().numpy()
    assert mask.shape == predict.shape, "mask.shape != predict.shape"
    AUC_ROC = roc_auc_score(mask, predict)

    return confmat.compute()[0], confmat.compute()[1], confmat.compute()[2], confmat.compute()[3], confmat.compute()[
        4], AUC_ROC


def train_one_epoch(model, optimizer, data_loader, device, epoch, total_epochs, scheduler,
                    scaler=None):
    model.train()
    total_loss = 0

    data_loader = tqdm(data_loader)
    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, False)
        total_loss += loss.item()

        data_loader.set_description(f"Epoch {epoch}/{total_epochs}")
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # chasedb1
        # scheduler.step()
    return total_loss / len(data_loader)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
