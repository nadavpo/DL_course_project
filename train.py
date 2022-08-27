import os

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
          val_loader, num_classes, best_metric, res_dir, n_epochs, mean, std, trigger, start_epoch, logger):
    for epoch in range(start_epoch, start_epoch + n_epochs + 1):
        mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scheduler,
                                    scaler=scaler)
        # drive
        # scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        acc, se, sp, F1, pr, AUC_ROC = evaluate(model, val_loader, device=device, num_classes=num_classes)

        print(f"train_loss: {mean_loss}, Precision: {pr}, acc: {acc}, F1-score: {F1}, AUC_ROC: {AUC_ROC}")
        print(f"se: {se}, sp: {sp}")

        losss.append(mean_loss)
        accs.append(acc)
        ses.append(se)
        sps.append(sp)
        f1s.append(F1)
        prs.append(pr)
        aucs.append(AUC_ROC)

        logger.report_scalar(title='loss', series='series', value=mean_loss, iteration=epoch)
        logger.report_scalar(title='acccuracy', series='series', value=acc, iteration=epoch)
        logger.report_scalar(title='F1 Score', series='series', value=F1, iteration=epoch)
        logger.report_scalar(title='AUC ROC', series='series', value=AUC_ROC, iteration=epoch)
        logger.report_scalar(title='precision', series='series', value=pr, iteration=epoch)
        logger.report_scalar(title='sp', series='series', value=sp, iteration=epoch)
        logger.report_scalar(title='se', series='series', value=se, iteration=epoch)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args,
                     "mean_im": mean,
                     "std_im": std}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        trigger += 1

        if AUC_ROC > best_metric["AUC_ROC"]:
            best_metric["AUC_ROC"] = AUC_ROC
            torch.save(save_file, os.path.join(res_dir, "best_model.pth"))
            trigger = 0
            print('saving model')
    return model, best_metric, losss, aucs, prs, f1s, sps, ses, accs


def main(args, task, logger):
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
                                               batch_size=8,
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

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

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

    res_dir = os.path.join('results', time.strftime("%d%m%Y_%H%M%S_") + args.exp_name)
    pathlib.Path(res_dir).mkdir(parents=True, exist_ok=True)

    model, best_metric, losss, aucs, prs, f1s, sps, ses, accs = train(args, model, optimizer, train_loader, device,
                                                                      scheduler, scaler, losss, aucs, prs, f1s, sps,
                                                                      ses, accs, val_loader, num_classes, best_metric,
                                                                      res_dir, args.epochs, mean, std, trigger,
                                                                      args.start_epoch, logger)
    if args.fine_tune_path:
        mean, std = compute_mean_std.compute(img_dir=os.path.join(args.fine_tune_path, 'images'))
        # logger.report_single_value('fine_tune_main', mean)
        # logger.report_single_value('fine_tune_main', std)
        # DRIVE
        train_dataset = DriveDataset(args.fine_tune_path,
                                     train=True,
                                     transforms=get_transform(train=True, mean=mean, std=std))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=8,
                                                   num_workers=num_workers,
                                                   shuffle=True,
                                                   pin_memory=True)
        model, best_metric, losss, aucs, prs, f1s, sps, ses, accs = train(args, model, optimizer, train_loader, device,
                                                                          scheduler, scaler, losss, aucs, prs, f1s, sps,
                                                                          ses, accs, val_loader, num_classes,
                                                                          best_metric,
                                                                          res_dir, args.start_epoch, mean, std, trigger,
                                                                          30, logger)

    with open(os.path.join(res_dir, 'losss.pkl'), 'wb') as f:
        pickle.dump(losss, f)
    with open(os.path.join(res_dir, 'accs.pkl'), 'wb') as f:
        pickle.dump(accs, f)
    with open(os.path.join(res_dir, 'ses.pkl'), 'wb') as f:
        pickle.dump(ses, f)
    with open(os.path.join(res_dir, 'sps.pkl'), 'wb') as f:
        pickle.dump(sps, f)
    with open(os.path.join(res_dir, 'f1s.pkl'), 'wb') as f:
        pickle.dump(f1s, f)
    with open(os.path.join(res_dir, 'prs.pkl'), 'wb') as f:
        pickle.dump(prs, f)
    with open(os.path.join(res_dir, 'aucs.pkl'), 'wb') as f:
        pickle.dump(aucs, f)

    task.upload_artifact('best model', artifact_object=os.path.join(res_dir, "best_model.pth"))
    print('finished')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch SA-UNET training")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--data-path", default="./DRIVE/aug_training")
    parser.add_argument("--fine-tune-path", default="./DRIVE/aug_training")
    parser.add_argument("--test-path", default="./DRIVE/test")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')

    # parser.add_argument('--resume', default=r'./results/24082022_235033_no_GAN/best_model.pth',
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--early_stop', default=35, type=int)

    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use pytorch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    task = Task.init(project_name="SA_UNET", task_name=args.exp_name)

    logger = Logger.current_logger()
    main(args, task, logger)
