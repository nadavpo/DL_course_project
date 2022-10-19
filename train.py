import os

import numpy as np
import torch
import torch.utils.data

import compute_mean_std
import transforms as T
from datasets import Chasedb1Datasets, DriveDataset
from model.SA_Unet import SA_UNet
from train_utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
import pickle
import time
import pathlib
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clearml import Task, Logger


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # min_size = int(0.5 * base_size)
        # max_size = int(1.2 * base_size)
        #
        # trans = [T.RandomResize(min_size, max_size)]

        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            # T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), flag=True):
    base_size = 565
    crop_size = 1008

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(crop_size, mean=mean, std=std)


def create_model(num_classes):
    model = SA_UNet(in_channels=3, num_classes=num_classes, base_c=16)
    return model


def train(args, model, optimizer, train_loader, device, scheduler, scaler, losss, aucs, prs, f1s, sps, ses, accs,
          val_loader, num_classes, best_metric, res_dir, n_epochs, mean, std, trigger, start_epoch, logger, lrs,
          metric_to_optim, is_fine_tune=False):
    for epoch in range(start_epoch, n_epochs + 1):
        mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, n_epochs,
                                    scheduler,
                                    scaler=scaler)
        # drive
        if scheduler:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"learning rate - {lr}\n")

        acc, se, sp, F1, pr, AUC_ROC = evaluate(model, val_loader, device=device, num_classes=num_classes)

        print(f"train_loss: {mean_loss}, Precision: {pr}")
        print(f"F1-score: {F1}, Accuracy: {acc}, AUC_ROC: {AUC_ROC}")
        print(f"Sensitivity: {se}, Specificity: {sp}")

        losss.append(mean_loss)
        accs.append(acc)
        ses.append(se)
        sps.append(sp)
        f1s.append(F1)
        prs.append(pr)
        aucs.append(AUC_ROC)
        lrs.append(lr)

        if logger:
            logger.report_scalar(title='loss', series='series', value=mean_loss, iteration=epoch)
            logger.report_scalar(title='Accuracy', series='Accuracy', value=acc, iteration=epoch)
            logger.report_scalar(title='F1 Score', series='F1 Score', value=F1, iteration=epoch)
            logger.report_scalar(title='AUC ROC', series='AUC ROC', value=AUC_ROC, iteration=epoch)
            logger.report_scalar(title='Precision', series='Precision', value=pr, iteration=epoch)
            logger.report_scalar(title='Specificity', series='Specificity', value=sp, iteration=epoch)
            logger.report_scalar(title='Sensitivity', series='Sensitivity', value=se, iteration=epoch)
            logger.report_scalar(title='lr', series='lr', value=lr, iteration=epoch)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": scheduler.state_dict() if scheduler else None,
                     "epoch": epoch,
                     "args": args,
                     "mean_im": mean,
                     "std_im": std}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        trigger += 1

        curr_res = {'AUC_ROC': AUC_ROC, "loss": mean_loss, "Accuracy": acc, "Sensitivity": se, "Specificity": sp,
                    "F1": F1, "Precision": pr}

        if curr_res[metric_to_optim] > best_metric[metric_to_optim]:
            best_metric["AUC_ROC"] = AUC_ROC
            best_metric["loss"] = mean_loss
            best_metric["Accuracy"] = acc
            best_metric["Sensitivity"] = se
            best_metric["Specificity"] = sp
            best_metric["F1"] = F1
            best_metric["Precision"] = pr
            if is_fine_tune:
                torch.save(save_file, os.path.join(res_dir, "best_model_fine_tune.pth"))
            else:
                torch.save(save_file, os.path.join(res_dir, "best_model.pth"))
            trigger = 0
            print('saving model')
    return model, best_metric, losss, aucs, prs, f1s, sps, ses, accs, lrs


def reset_optimizer(params_to_optimize, train_loader, epochs):
    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr
    )

    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.65 ** epoch)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    # more options https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
    # scheduler = None
    return optimizer, scheduler


def print_WP(f1s, aucs, accs, prs, sps, ses, ind, f):
    print(f"Precision: {prs[ind]}, F1-score: {f1s[ind]}")
    print(f"Accuracy: {accs[ind]}, AUC_ROC: {aucs[ind]}")
    print(f"Sensitivity: {ses[ind]}, Specificity: {sps[ind]}")
    f.write(f"Precision: {prs[ind]}\n")
    f.write(f"F1-score: {f1s[ind]}\n")
    f.write(f"Accuracy: {accs[ind]}\n")
    f.write(f"AUC_ROC: {aucs[ind]}\n")
    f.write(f"Sensitivity: {ses[ind]}\n")
    f.write(f"Specificity: {sps[ind]}\n")
    f.write(f"\n\n")


def print_save_metrics(f1s, aucs, accs, prs, sps, ses, res_dir, postfix=""):
    with open(os.path.join(res_dir, 'best metrics' + postfix +'.txt'), 'w') as f:
        print(f"best result F1:")
        f.write(f"best result F1:\n")
        ind = f1s.index(max(f1s))
        print_WP(f1s, aucs, accs, prs, sps, ses, ind, f)

        print(f"best result Accuracy:")
        f.write(f"best result Accuracy:\n")
        ind = accs.index(max(accs))
        print_WP(f1s, aucs, accs, prs, sps, ses, ind, f)

        print(f"best result AUC_ROC:")
        f.write(f"best result AUC_ROC:\n")
        ind = aucs.index(max(aucs))
        print_WP(f1s, aucs, accs, prs, sps, ses, ind, f)


def main(args, task=None, logger=None):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes

    # using compute_mean_std.py

    mean, std = compute_mean_std.compute(img_dir=os.path.join(args.data_path, 'images'))

    # logger.report_single_value('mean_main', mean)
    # logger.report_single_value('std_main', std)
    # DRIVE
    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 num_data_exp=args.num_data_exp,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.test_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    # ChaseDB1
    # train_dataset = Chasedb1Datasets(args.data_path,
    #                                  train=True,
    #                                  transforms=get_transform(train=True, mean=mean, std=std))
    #
    # val_dataset = Chasedb1Datasets(args.data_path,
    #                                train=False,
    #                                transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    optimizer, scheduler = reset_optimizer(params_to_optimize, train_loader, args.epochs)

    if args.resume:
        print('resuming from file')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    trigger = 0
    best_metric = {"AUC_ROC": 0.5}
    losss = []
    aucs = []
    prs = []
    f1s = []
    sps = []
    ses = []
    accs = []
    lrs = []

    res_dir = os.path.join('results', time.strftime("%d%m%Y_%H%M%S_") + args.exp_name)
    pathlib.Path(res_dir).mkdir(parents=True, exist_ok=True)

    model, best_metric, losss, aucs, prs, f1s, sps, ses, accs, lrs = train(args, model, optimizer, train_loader, device,
                                                                           scheduler, scaler, losss, aucs, prs, f1s,
                                                                           sps,
                                                                           ses, accs, val_loader, num_classes,
                                                                           best_metric,
                                                                           res_dir, args.epochs, mean, std, trigger,
                                                                           args.start_epoch, logger, lrs,
                                                                           args.metric_to_optim)
    save_articat(losss, 'losss', 'loss', res_dir)
    save_articat(accs, 'Accuracies', 'Accuracy', res_dir)
    save_articat(ses, 'Sensitivities', 'Sensitivity', res_dir)
    save_articat(sps, 'Specificities', 'Specificity', res_dir)
    save_articat(f1s, 'F1 Scores', 'F1 score', res_dir)
    save_articat(prs, 'Precision', 'Precision', res_dir)
    save_articat(aucs, 'AUC_ROC', 'AUC_ROC', res_dir)
    save_articat(lrs, 'LR', 'LR', res_dir)

    print_save_metrics(f1s, aucs, accs, prs, sps, ses, res_dir)

    if args.fine_tune_path and args.fine_tune_epochs > 0:
        print("start fine tune")
        mean, std = compute_mean_std.compute(img_dir=os.path.join(args.fine_tune_path, 'images'))

        optimizer, scheduler = reset_optimizer(params_to_optimize, train_loader, args.fine_tune_epochs)

        # logger.report_single_value('fine_tune_main', mean)
        # logger.report_single_value('fine_tune_main', std)
        # DRIVE
        train_dataset = DriveDataset(args.fine_tune_path,
                                     train=True,
                                     num_data_exp=args.num_fine_tune_exp,
                                     transforms=get_transform(train=True, mean=mean, std=std))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True)
        model, best_metric, losss, aucs, prs, f1s, sps, ses, accs, lrs = train(args, model, optimizer, train_loader,
                                                                               device,
                                                                               None, scaler, losss, aucs, prs, f1s, sps,
                                                                               ses, accs, val_loader, num_classes,
                                                                               best_metric,
                                                                               res_dir,
                                                                               args.epochs + args.fine_tune_epochs,
                                                                               mean, std, trigger,
                                                                               args.epochs + 1, logger, lrs,
                                                                               args.metric_to_optim, is_fine_tune=True)
        save_articat(losss, 'losss - fine tune', 'loss', res_dir)
        save_articat(accs, 'Accuracies - fine tune', 'Accuracy', res_dir)
        save_articat(ses, 'Sensitivities - fine tune', 'Sensitivity', res_dir)
        save_articat(sps, 'Specificities - fine tune', 'Specificity', res_dir)
        save_articat(f1s, 'F1 Scores - fine tune', 'F1 score', res_dir)
        save_articat(prs, 'Precision - fine tune', 'Precision', res_dir)
        save_articat(aucs, 'AUC_ROC - fine tune', 'AUC_ROC', res_dir)
        save_articat(lrs, 'LR - fine tune', 'LR', res_dir)

        print_save_metrics(f1s, aucs, accs, prs, sps, ses, res_dir, postfix='_fine_tune')

    if task:
        task.upload_artifact('best model', artifact_object=os.path.join(res_dir, "best_model.pth"))
    print('finished')


def save_articat(data, name, y_label, res_dir):
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.title(name)
    plt.xlabel('epoch #')
    plt.ylabel(y_label)
    plt.savefig(os.path.join(res_dir, name + '.png'))

    with open(os.path.join(res_dir, name + '.pkl'), 'wb') as f:
        pickle.dump(data, f)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch SA-UNET training")
    parser.add_argument("--exp_name", default='orig train WITH augs, pre infer on gan AUC_ROC, 200 epochs')

    parser.add_argument("--data_path", default="./DRIVE/training_aug")
    # parser.add_argument("--data_path", default="./DRIVE/training")
    parser.add_argument("--num_data_exp", default=-1, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N")

    parser.add_argument("--fine_tune_path", default="")
    # parser.add_argument("--fine_tune_path", default="./DRIVE/gan_data_new_aug")
    parser.add_argument("--num_fine_tune_exp", default=-1, type=int)
    parser.add_argument("--fine_tune_epochs", default=100, type=int, metavar="N")

    parser.add_argument("--test_path", default="./DRIVE/test")
    # exclude background
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=2, type=int)
    parser.add_argument("--metric_to_optim", default="AUC_ROC")  # AUC_ROC,Accuracy,F1,Precision,Sensitivity,Specificity
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

    # parser.add_argument('--resume', default=r'./results/24082022_235033_no_GAN/best_model.pth',
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--early_stop', default=35, type=int)

    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use pytorch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    run_clearml = True
    if run_clearml:
        task = Task.init(project_name="SA_UNET", task_name=args.exp_name)
        logger = Logger.current_logger()
    else:
        task = None
        logger = None

    main(args, task, logger)
