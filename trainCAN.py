import os
import torch
import wandb
import random
import argparse
import datetime
import pandas as pd
import torch.optim as optim
import torch.distributed as dist

from tqdm import tqdm
from CAN.model import CAN
from sklearn import metrics
from utils.init import init_seeds
from utils.augment import get_augs
from torch.utils.data import DataLoader
from dataset.test_dataset import Test_Dataset
from utils.loss import ConsistencyCos, PolyBCELoss
from timm.scheduler.cosine_lr import CosineLRScheduler
from dataset.train_dataset import FakeFaceDataset, RealFaceDataset
from utils.tools import expand_prediction, AverageMeter,SequentialDistributedSampler, distributed_concat


#wandb.ai projectname
project_name =  "CAN"
model_name = "CAN"
OUTPUT_DIR = "weights/" + "CAN"

config_defaults = {
    "epochs": 30,
    "train_batch_size": 8,
    "valid_batch_size": 12,
    "gpu_nums": 1,
    "optimizer": "adam",
    "learning_rate": 2e-4,
    "weight_decay": 0,
    "rand_seed": 888,
    "accumulation_steps": 2,
}


def train(name, train_df, val_df, test_df):

    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument("--local_rank", type=int, default=1)
    args = parser.parse_args()
    local_rank = args.local_rank
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run = f"{name}_"
    print("Starting -->", run)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))
    local_rank = torch.distributed.get_rank()
    device=torch.device(local_rank)
    init_seeds(0 + local_rank)

    wandb.init(project=project_name, config=config_defaults, name=run)
    config = wandb.config

    model = CAN()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    COS_Loss = ConsistencyCos(device).to(device)
    Poly_BCE_Loss = PolyBCELoss(epsilon=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))

    lr_scheduler = CosineLRScheduler(optimizer, t_initial=int(config.epochs//2), lr_min=1e-6)

    data_train_real = RealFaceDataset(
        mode="train",
        df=train_df,
        transforms=get_augs(name="MLFM_High",norm='imagenet',size=(320, 320)),
    )
    
    train_sampler_real = torch.utils.data.distributed.DistributedSampler(data_train_real)

    train_data_loader_real = DataLoader(
        data_train_real, 
        batch_size=int(config.train_batch_size//config.gpu_nums//2), 
        num_workers=8, 
        shuffle=False, 
        drop_last=True, 
        sampler=train_sampler_real,
        pin_memory=True)


    data_train_fake = FakeFaceDataset(
        mode="train",
        df=train_df,
        transforms=get_augs(name="MLFM_High",norm='imagenet',size=(320, 320)),
    )
    
    train_sampler_fake = torch.utils.data.distributed.DistributedSampler(data_train_fake)

    train_data_loader_fake= DataLoader(
        data_train_fake, 
        batch_size=int(config.train_batch_size//config.gpu_nums//2), 
        num_workers=8, 
        shuffle=False, 
        drop_last=True,
        sampler=train_sampler_fake,
        pin_memory=True)
    
    for epoch in range(0, config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")
        train_data_loader_real.sampler.set_epoch(epoch)
        train_data_loader_fake.sampler.set_epoch(epoch)

        train_epoch(config, model, train_data_loader_real, train_data_loader_fake, optimizer, Poly_BCE_Loss, COS_Loss, epoch, device)
        lr_scheduler.step(epoch)
        test_epoch(model, test_df, optimizer, lr_scheduler, epoch, local_rank, device)


def train_epoch(config, model, train_data_loader_real, train_data_loader_fake, optimizer, Poly_BCE_Loss, COS_Loss, epoch, device):

    train_loss = AverageMeter()
    len_ = train_data_loader_real.__len__()
    real_iter = iter(train_data_loader_real)
    fake_iter = iter(train_data_loader_fake)
    
    for i in tqdm(range(len_)):
        model.train()
        try:
            data_real = real_iter.next()
            data_fake = fake_iter.next()
        except StopIteration:
            break

        batch_images_real = data_real["image"].detach().to(device)
        batch_images_fake = data_fake["image"].detach().to(device)
        batch_labels_real = data_real["label"].detach().to(device)
        batch_labels_fake = data_fake["label"].detach().to(device)

        batch_refimage_real = data_real["ref_image"].detach().to(device)
        batch_refimage_fake = data_fake["ref_image"].detach().to(device)
        
        _im = torch.cat((batch_images_real, batch_images_fake), dim=0)
        _ref = torch.cat((batch_refimage_real, batch_refimage_fake), dim=0)
        _lab = torch.cat((batch_labels_real, batch_labels_fake), dim=0)
        
        idx_ = list(range(_im.shape[0]))
        random.shuffle(idx_)
        _im = _im[idx_]
        _ref = _ref[idx_]
        _lab = _lab[idx_]
        
        batch_input = torch.cat((_im, _ref), dim=0)
        batch_labels = torch.cat((_lab, _lab), dim=0)

        result = model(batch_input)
        fea, out = result['features'], result['logits']

        loss = Poly_BCE_Loss(out, batch_labels.view(-1, 1).type_as(out)) + 2 * COS_Loss(fea)
        loss = loss/config.accumulation_steps
        loss.backward()

        if ((i+1) % config.accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss.update(loss.item(), train_data_loader_real.batch_size*2)

        if i % 100 == 0 and dist.get_rank()==0:
            train_metrics = {
                "train_loss": train_loss.avg,
                "epoch": epoch,
            }
            wandb.log(train_metrics)


def test_epoch(model, test_df, optimizer, lr_scheduler, epoch, device):
    model.eval()
    data = Test_Dataset(
            test_df, 
            get_augs(name="Test",norm='imagenet',size=(320, 320))
        )
    data_sampler = SequentialDistributedSampler(data, batch_size=24)
    data_loader = DataLoader(
        data,
        batch_size=24,
        num_workers=8,
        shuffle=False,
        drop_last=True,
        sampler=data_sampler,
        pin_memory=True
    )

    with torch.no_grad():
        predictions = []
        targets = []
        for batch in tqdm(data_loader):
            batch_images = batch["image"].to(device)
            batch_labels = batch["label"].to(device)
            result = model(batch_images, batch_images)
            fea, out = result['features'], result['logits']
            batch_targets = (batch_labels.view(-1, 1) >= 0.5) * 1
            batch_preds = torch.sigmoid(out)

            targets.append(batch_targets)
            predictions.append(batch_preds.detach())

        torch.distributed.barrier()
        targets = distributed_concat(torch.cat(targets, dim=0), len(data_sampler.dataset)).cpu()
        predictions = distributed_concat(torch.cat(predictions, dim=0), len(data_sampler.dataset)).cpu()

        acc = metrics.accuracy_score(targets, (predictions >= 0.5) * 1)
        auc = metrics.roc_auc_score(targets, predictions)
        mAP = metrics.average_precision_score(targets, predictions)
        log_loss = metrics.log_loss(targets, expand_prediction(predictions))
        torch.distributed.barrier()


        if dist.get_rank()==0:
            test_metrics = {
                "test_Logloss": log_loss,
                "test_auc": auc,
                "test_acc": acc,
                "test_mAP": mAP,
            }
            wandb.log(test_metrics)

            checkpoint = {
                "epoch": epoch,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(OUTPUT_DIR, str(epoch) + '_'+ str(auc) +'.h5'))
        torch.distributed.barrier()
 
if __name__ == "__main__":

    train_df = pd.read_csv('/data/jinghui.sun/B4_320/DATA/FF_270_train_NoMask.csv')
    val_df = pd.read_csv('/data/jinghui.sun/B4_320/DATA/FF_270_valid_NoMask.csv')
    test_df = pd.read_csv('/data/jinghui.sun/B4_320/DATA/FF_50_test_NoMask.csv')
    train(name="Combined_hardcore_" + model_name, train_df=train_df, val_df=val_df, test_df=test_df)
    



