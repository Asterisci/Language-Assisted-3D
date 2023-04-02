from email.policy import strict
import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnet import RefNet

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train_parser.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val_parser.json")))
# NR3D_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train_attr.json")))
# NR3D_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val_attr.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, config, augment, dist):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[split], 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        color_prediction=args.color_prediction,
        relation_prediction=args.relation_prediction,
        shape_prediction=args.shape_prediction,
        size_prediction=args.size_prediction,
        sampled_box=args.sampled_box,
    )
    if dist:
        data_sampler = DistributedSampler(dataset, shuffle=split=='train', seed=args.seed)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, sampler=data_sampler, pin_memory=True)
    else:
        data_sampler = None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=split=='train', num_workers=0, pin_memory=True)

    return dataset, dataloader, data_sampler

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        relation_prediction=args.relation_prediction,
        prepare_epoch=args.prepare_epoch,
        color_prediction=args.color_prediction,
        shape_prediction=args.shape_prediction,
        size_prediction=args.size_prediction,
        MLCVNet_backbone=args.MLCV
    )

        # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained VoteNet...")
        input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
        pretrained_model = RefNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_bidir=args.use_bidir,
            no_reference=True,
            relation_prediction=True,
            prepare_epoch=args.prepare_epoch,
            color_prediction=args.color_prediction,
            shape_prediction=args.shape_prediction,
            size_prediction=args.size_prediction,
            MLCVNet_backbone=args.MLCV
        )

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal
        model.lang = pretrained_model.lang

        if args.no_detection:
            # freeze pointnet++ backbone
            for param in model.backbone_net.parameters():
                param.requires_grad = False

            # freeze voting
            for param in model.vgen.parameters():
                param.requires_grad = False
            
            # freeze detector
            for param in model.proposal.parameters():
                param.requires_grad = False

    
    # to CUDA
    if args.local_rank != -1:
        device = torch.device('cuda', args.local_rank)
        model = model.cuda(device)
    else:
        model = model.cuda()

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True, broadcast_buffers=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)

        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"), map_location=torch.device('cpu'))
        
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [40, 50]
    LR_DECAY_RATE = 0.3
    BN_DECAY_STEP = None
    BN_DECAY_RATE = None

    solver = Solver(
        model=model, 
        config=DC, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        local_rank=args.local_rank,
        val_step=args.val_step,
        prepare_epoch=args.prepare_epoch,
        detection=not args.no_detection,
        reference=not args.no_reference, 
        use_lang_classifier=not args.no_lang_cls,
        relation_prediction=args.relation_prediction,
        color_prediction=args.color_prediction,
        shape_prediction=args.shape_prediction,
        size_prediction=args.size_prediction,
        fully_sup=args.fully_sup,
        val_dist=args.val_dist,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        MLCVNet_backbone=args.MLCV
    )
    num_params = get_num_params(model)

    return solver, num_params, root

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):
    if args.no_reference:
    #     train_scene_list = get_scannet_scene_list("train")
    #     new_scanrefer_train = []
    #     for scene_id in train_scene_list:
    #         data = deepcopy(SCANREFER_TRAIN[0])
    #         data["scene_id"] = scene_id
    #         new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
    # get initial scene list
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        new_scanrefer_val = scanrefer_val

    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    if num_scenes == -1: 
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
    
    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list

def train(args):
    # init training dataset
    print("preparing data...")
    TRAIN = NR3D_TRAIN if args.nr3d else SCANREFER_TRAIN
    VAL = NR3D_VAL if args.nr3d else SCANREFER_VAL
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(TRAIN, VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }

    # dataloader
    train_dataset, train_dataloader, train_sample = get_dataloader(args, scanrefer, all_scene_list, "train", DC, True, args.local_rank != -1)
    val_dataset, val_dataloader, val_sample = get_dataloader(args, scanrefer, all_scene_list, "val", DC, False, args.local_rank != -1 and args.val_dist)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader,
        "train_sample": train_sample,
        "val_sample": val_sample
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=12)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--relation_prediction", action="store_true", help="relation prediction task.")
    parser.add_argument("--prepare_epoch", type=float, help="prepare epoch for relation prediction task.", default=0)
    parser.add_argument("--color_prediction", action="store_true", help="color prediction task.")
    parser.add_argument("--size_prediction", action="store_true", help="size prediction task.")
    parser.add_argument("--shape_prediction", action="store_true", help="shape prediction task.")
    parser.add_argument("--fully_sup", action="store_true", help="fully supervised.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--val_dist", action="store_true", help="use distributed val.")
    parser.add_argument("--MLCV", action="store_true", help="use MLCVNet backbone.")
    parser.add_argument("--nr3d", action="store_true", help="use Nr3d dataset.")
    parser.add_argument("--sampled_box", type=int, default=10000)

    args = parser.parse_args()

    if args.local_rank == -1:
        # setting
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        # dist
        torch.cuda.set_device(args.local_rank)
        # init dist
        dist.init_process_group(backend='nccl')
        dist.barrier()

    # reproducibility
    seed = args.seed
    print("seed {}...".format(seed))
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    train(args)
    
