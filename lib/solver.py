import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR


sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from utils.eta import decode_eta
from lib.pointnet2.pytorch_utils import BNMomentumScheduler
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths

from lib.dist_utils import reduce_tensor


ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_ref_loss: {train_ref_loss}
[loss] train_lang_loss: {train_lang_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_vote_loss: {train_vote_loss}
[loss] train_box_loss: {train_box_loss}
[loss] train_relation_loss: {train_relation_loss}
[loss] train_color_loss: {train_color_loss}
[loss] train_shape_loss: {train_shape_loss}
[loss] train_size_loss: {train_size_loss}
[loss] train_lang_acc: {train_lang_acc}
[sco.] train_ref_acc: {train_ref_acc}
[sco.] train_obj_acc: {train_obj_acc}
[sco.] train_sem_acc: {train_sem_acc}
[sco.] train_relation_acc: {train_relation_acc}
[sco.] train_color_acc: {train_color_acc}
[sco.] train_shape_acc: {train_shape_acc}
[sco.] train_size_acc: {train_size_acc}
[sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[sco.] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_ref_loss: {train_ref_loss}
[train] train_lang_loss: {train_lang_loss}
[train] train_objectness_loss: {train_objectness_loss}
[train] train_vote_loss: {train_vote_loss}
[train] train_box_loss: {train_box_loss}
[train] train_relation_loss: {train_relation_loss}
[train] train_color_loss: {train_color_loss}
[train] train_shape_loss: {train_shape_loss}
[train] train_size_loss: {train_size_loss}
[train] train_lang_acc: {train_lang_acc}
[train] train_ref_acc: {train_ref_acc}
[train] train_obj_acc: {train_obj_acc}
[train] train_sem_acc: {train_sem_acc}
[train] train_relation_acc: {train_relation_acc}
[train] train_color_acc: {train_color_acc}
[train] train_shape_acc: {train_shape_acc}
[train] train_size_acc: {train_size_acc}
[train] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[train] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[val]   val_loss: {val_loss}
[val]   val_ref_loss: {val_ref_loss}
[val]   val_lang_loss: {val_lang_loss}
[val]   val_objectness_loss: {val_objectness_loss}
[val]   val_vote_loss: {val_vote_loss}
[val]   val_box_loss: {val_box_loss}
[val]   val_relation_loss: {val_relation_loss}
[val]   val_color_loss: {val_color_loss}
[val]   val_shape_loss: {val_shape_loss}
[val]   val_size_loss: {val_size_loss}
[val]   val_lang_acc: {val_lang_acc}
[val]   val_ref_acc: {val_ref_acc}
[val]   val_obj_acc: {val_obj_acc}
[val]   val_sem_acc: {val_sem_acc}
[val]   val_relation_acc: {val_relation_acc}
[val]   val_color_acc: {val_color_acc}
[val]   val_shape_acc: {val_shape_acc}
[val]   val_size_acc: {val_size_acc}
[val]   val_pos_ratio: {val_pos_ratio}, val_neg_ratio: {val_neg_ratio}
[val]   val_iou_rate_0.25: {val_iou_rate_25}, val_iou_rate_0.5: {val_iou_rate_5}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] ref_loss: {ref_loss}
[loss] lang_loss: {lang_loss}
[loss] objectness_loss: {objectness_loss}
[loss] vote_loss: {vote_loss}
[loss] box_loss: {box_loss}
[loss] relation_loss: {relation_loss}
[loss] color_loss: {color_loss}
[loss] shape_loss: {shape_loss}
[loss] size_loss: {size_loss}
[loss] lang_acc: {lang_acc}
[sco.] ref_acc: {ref_acc}
[sco.] obj_acc: {obj_acc}
[sco.] sem_acc: {sem_acc}
[sco.] relation_acc: {relation_acc}
[sco.] color_acc: {color_acc}
[sco.] shape_acc: {shape_acc}
[sco.] size_acc: {size_acc}
[sco.] mAP_0.25: {map_25}
[sco.] mAP_0.5: {map_5}
[sco.] pos_ratio: {pos_ratio}, neg_ratio: {neg_ratio}
[sco.] iou_rate_0.25: {iou_rate_25}, iou_rate_0.5: {iou_rate_5}
"""

class Solver():
    def __init__(self, model, config, dataloader, optimizer, stamp, local_rank, val_step=10, prepare_epoch=0, val_dist=False,
    detection=True, reference=True, use_lang_classifier=True, 
    relation_prediction=False, color_prediction=False, shape_prediction=False, size_prediction=False,
    fully_sup=False,
    lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None, MLCVNet_backbone=False):

        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step * len(self.dataloader["train"])
        self.local_rank = local_rank
        self.prepare_epoch = prepare_epoch

        self.detection = detection
        self.reference = reference
        self.use_lang_classifier = use_lang_classifier
        self.relation_prediction = relation_prediction
        self.color_prediction = color_prediction
        self.shape_prediction = shape_prediction
        self.size_prediction = size_prediction
        self.fully_sup = fully_sup
        self.val_dist = val_dist
        self.pretask = relation_prediction or color_prediction or shape_prediction or size_prediction

        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "ref_loss": float("inf"),
            "lang_loss": float("inf"),
            "objectness_loss": float("inf"),
            "vote_loss": float("inf"),
            "box_loss": float("inf"),
            "relation_loss": float("inf"),
            "color_loss": float("inf"),
            "shape_loss": float("inf"),
            "size_loss": float("inf"),
            "lang_acc": -float("inf"),
            "ref_acc": -float("inf"),
            "obj_acc": -float("inf"),
            "pos_ratio": -float("inf"),
            "neg_ratio": -float("inf"),
            "iou_rate_0.25": -float("inf"),
            "iou_rate_0.5": -float("inf"),
            "sem_acc": -float("inf"),
            "relation_acc": -float("inf"),
            "color_acc": -float("inf"),
            "shape_acc": -float("inf"),
            "size_acc": -float("inf"),
            "mAP_0.25": -float("inf"),
            "mAP_0.5": -float("inf"),
        }

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }
        
        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if lr_decay_step and lr_decay_rate:
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate**(int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)
        else:
            self.bn_scheduler = None

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * self.val_step
        
        for epoch_id in range(epoch):
            try:
                if self.local_rank != -1:
                    self.dataloader['train_sample'].set_epoch(epoch_id)
                self._log("epoch {} starting...".format(epoch_id + 1))

                # feed 
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                # self._log("saving last models...\n")
                # model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                # torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))
                if self.local_rank in [0, -1]:
                    # save model
                    self._log("saving last models...\n")
                    model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                    if self.local_rank == 0:
                        torch.save(self.model.module.state_dict(), os.path.join(model_root, "model_last.pth"))
                    else:
                        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))
                           

                # update lr scheduler
                if self.lr_scheduler:
                    print("update learning rate --> {}\n".format(self.lr_scheduler.get_last_lr()))
                    self.lr_scheduler.step()

                # update bn scheduler
                if self.bn_scheduler:
                    print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                    self.bn_scheduler.step()
                
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        # self.log_fout.write(info_str + "\n")
        # self.log_fout.flush()
        # print(info_str)
        if self.local_rank in [0, -1]:
             self.log_fout.write(info_str + "\n")
             self.log_fout.flush()
             print(info_str)

    def _reset_log(self, phase):
        self.log[phase] = {
            # info
            "forward": [],
            "backward": [],
            "eval": [],
            "fetch": [],
            "iter_time": [],
            # loss (float, not torch.cuda.FloatTensor)
            "loss": [],
            "ref_loss": [],
            "lang_loss": [],
            "objectness_loss": [],
            "vote_loss": [],
            "box_loss": [],
            "relation_loss": [],
            "color_loss": [],
            "shape_loss": [],
            "size_loss": [],
            # scores (float, not torch.cuda.FloatTensor)
            "lang_acc": [],
            "ref_acc": [],
            "obj_acc": [],
            "relation_acc": [],
            "color_acc": [],
            "shape_acc": [],
            "size_acc": [],
            "pos_ratio": [],
            "neg_ratio": [],
            "iou_rate_0.25": [],
            "iou_rate_0.5": [],
            # detection
            "sem_acc": []
        }

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict, epoch_id, dist=True):
        if not dist:
            data_dict = self.model.module(data_dict, epoch_id)
        else:
            data_dict = self.model(data_dict, epoch_id)
        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict, epoch_id, reduce):
        _, data_dict = get_loss(
            data_dict=data_dict, 
            config=self.config, 
            detection=self.detection,
            reference=self.reference, 
            use_lang_classifier=self.use_lang_classifier,
            relation_prediction=self.relation_prediction,
            color_prediction=self.color_prediction,
            shape_prediction=self.shape_prediction,
            size_prediction=self.size_prediction,
            epoch_id=epoch_id,
            prepare_epoch=self.prepare_epoch,
            fully_sup=self.fully_sup,
        )

        if reduce:
            reduce_list = [
                "ref_loss",
                "lang_loss",
                "objectness_loss",
                "vote_loss",
                "box_loss",
                "relation_loss",
                "color_loss",
                "shape_loss",
                "size_loss",
                "loss"
            ]
            for i in reduce_list:
                data_dict[i] = reduce_tensor(data_dict[i])

        # dump
        self._running_log["ref_loss"] = data_dict["ref_loss"]
        self._running_log["lang_loss"] = data_dict["lang_loss"]
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["vote_loss"] = data_dict["vote_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["relation_loss"] = data_dict["relation_loss"]
        self._running_log["color_loss"] = data_dict["color_loss"]
        self._running_log["shape_loss"] = data_dict["shape_loss"]
        self._running_log["size_loss"] = data_dict["size_loss"]
        self._running_log["loss"] = data_dict["loss"]

    def _eval(self, data_dict, phase, reduce, epoch_id):
        if not self.reference and phase == 'val':
            data_dict = get_eval(
                data_dict=data_dict,
                config=self.config,
                reference=self.reference,
                post_processing=self.POST_DICT,
                relation_prediction=self.relation_prediction,
                color_prediction=self.color_prediction,
                shape_prediction=self.shape_prediction,
                size_prediction=self.size_prediction,
                prepare=epoch_id<self.prepare_epoch
            )
            batch_pred_map_cls = parse_predictions(data_dict, self.POST_DICT) 
            batch_gt_map_cls = parse_groundtruths(data_dict,self.POST_DICT) 
            for ap_calculator in self.AP_CALCULATOR_LIST:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        else:
            data_dict = get_eval(
                data_dict=data_dict,
                config=self.config,
                reference=self.reference,
                use_lang_classifier=self.use_lang_classifier,
                relation_prediction=self.relation_prediction,
                color_prediction=self.color_prediction,
                shape_prediction=self.shape_prediction,
                size_prediction=self.size_prediction,
                prepare=epoch_id<self.prepare_epoch
            )

        if reduce:
            reduce_list = [
                "lang_acc",
                "ref_acc",
                "obj_acc",
                "pos_ratio",
                "neg_ratio",
                "ref_iou_rate_0.25",
                "ref_iou_rate_0.5",
                "sem_acc",
                "relation_acc",
                "color_acc",
                "shape_acc",
                "size_acc",
            ]
            for i in reduce_list:
                data_dict[i] = reduce_tensor(data_dict[i])
            self._running_log["ref_acc"] = data_dict["ref_acc"].mean().item()
            self._running_log["iou_rate_0.25"] = data_dict["ref_iou_rate_0.25"].mean().item()
            self._running_log["iou_rate_0.5"] = data_dict["ref_iou_rate_0.5"].mean().item()
        else:
            self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
            self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
            self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])

        # dump
        self._running_log["lang_acc"] = data_dict["lang_acc"].item()
        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["relation_acc"] = data_dict["relation_acc"].item()
        self._running_log["color_acc"] = data_dict["color_acc"].item()
        self._running_log["shape_acc"] = data_dict["shape_acc"].item()
        self._running_log["size_acc"] = data_dict["size_acc"].item()
        self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()
        self._running_log["sem_acc"] = data_dict["sem_acc"].item()


    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)

        # re-init log
        self._reset_log(phase)

        if not self.reference and phase == "val":
            self.POST_DICT = {
                "remove_empty_box": True, 
                "use_3d_nms": True, 
                "nms_iou": 0.25,
                "use_old_type_nms": False, 
                "cls_nms": True, 
                "per_class_proposal": True,
                "conf_thresh": 0.05,
                "dataset_config": self.config
            }
            self.AP_IOU_THRESHOLDS = [0.25, 0.5]
            self.AP_CALCULATOR_LIST = [APCalculator(iou_thresh, self.config.class2type) for iou_thresh in self.AP_IOU_THRESHOLDS]

        # change dataloader
        dataloader = dataloader if phase == "train" else tqdm(dataloader)

        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                data_dict[key] = data_dict[key].cuda()

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "ref_loss": 0,
                "lang_loss": 0,
                "objectness_loss": 0,
                "vote_loss": 0,
                "box_loss": 0,
                "relation_loss": 0,
                "color_loss": 0,
                "shape_loss": 0,
                "size_loss": 0,
                # acc
                "lang_acc": 0,
                "ref_acc": 0,
                "obj_acc": 0,
                "pos_ratio": 0,
                "neg_ratio": 0,
                "iou_rate_0.25": 0,
                "iou_rate_0.5": 0,
                "sem_acc": 0,
                "relation_acc": 0,
                "color_acc": 0,
                "shape_acc": 0,
                "size_acc": 0,
            }

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            with torch.autograd.set_detect_anomaly(True) if phase == "train" else torch.no_grad():
                # forward
                start = time.time()
                data_dict = self._forward(data_dict, epoch_id, self.val_dist or phase != "val" or self.local_rank == -1)
                self._compute_loss(data_dict, epoch_id, self.val_dist and phase == "val")
                self.log[phase]["forward"].append(time.time() - start)

                # backward
                if phase == "train":
                    start = time.time()
                    self._backward()
                    self.log[phase]["backward"].append(time.time() - start)
            
            with torch.no_grad():

                # eval
                start = time.time()
                self._eval(data_dict, phase, self.val_dist and phase == "val", epoch_id)
                self.log[phase]["eval"].append(time.time() - start)

            # record log
            self.log[phase]["loss"].append(self._running_log["loss"].item())
            self.log[phase]["ref_loss"].append(self._running_log["ref_loss"].item())
            self.log[phase]["lang_loss"].append(self._running_log["lang_loss"].item())
            self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
            self.log[phase]["vote_loss"].append(self._running_log["vote_loss"].item())
            self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())
            self.log[phase]["relation_loss"].append(self._running_log["relation_loss"].item())
            self.log[phase]["color_loss"].append(self._running_log["color_loss"].item())
            self.log[phase]["shape_loss"].append(self._running_log["shape_loss"].item())
            self.log[phase]["size_loss"].append(self._running_log["size_loss"].item())

            self.log[phase]["lang_acc"].append(self._running_log["lang_acc"])
            self.log[phase]["ref_acc"].append(self._running_log["ref_acc"])
            self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
            self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
            self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])
            self.log[phase]["iou_rate_0.25"].append(self._running_log["iou_rate_0.25"])
            self.log[phase]["iou_rate_0.5"].append(self._running_log["iou_rate_0.5"])    
            self.log[phase]["relation_acc"].append(self._running_log["relation_acc"])
            self.log[phase]["color_acc"].append(self._running_log["color_acc"])     
            self.log[phase]["shape_acc"].append(self._running_log["shape_acc"])     
            self.log[phase]["size_acc"].append(self._running_log["size_acc"])       
            self.log[phase]["sem_acc"].append(self._running_log["sem_acc"])     

            # report
            if phase == "train":
                iter_time = self.log[phase]["fetch"][-1]
                iter_time += self.log[phase]["forward"][-1]
                iter_time += self.log[phase]["backward"][-1]
                iter_time += self.log[phase]["eval"][-1]
                self.log[phase]["iter_time"].append(iter_time)
                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)

                # evaluation
                if self._global_iter_id % self.val_step == 0:
                    print("evaluating...")
                    if self.local_rank != -1 and self.val_dist:
                        self.dataloader['val_sample'].set_epoch(epoch_id)
                    # val
                    if self.local_rank not in [-1, 0] and not self.val_dist:
                        torch.distributed.barrier()
                    else:
                        self._feed(self.dataloader["val"], "val", epoch_id)
                        self._dump_log("val")
                        self._epoch_report(epoch_id)
                        if self.local_rank == 0 and not self.val_dist:
                            torch.distributed.barrier()

                    self._set_phase("train")
                # dump log
                self._dump_log("train")
                self._global_iter_id += 1

        if epoch_id == 40 and self.local_rank == 0:
            self._log("saving 40 checkpoint...\n")
            save_dict = {
                "epoch": epoch_id,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }
            checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
            torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint_40.tar"))


        # check best
        if phase == "val":
            if not self.reference:
                _metrics_dict = {}
                for i, ap_calculator in enumerate(self.AP_CALCULATOR_LIST):
                    metrics_dict = ap_calculator.compute_metrics()
                    self._log("-"*10+"iou_thresh: %f"%(self.AP_IOU_THRESHOLDS[i])+"-"*10)
                    for key in metrics_dict:
                        self._log("eval %s: %f"%(key, reduce_tensor(metrics_dict[key]).item() if self.val_dist else metrics_dict[key]))
                    _metrics_dict[self.AP_IOU_THRESHOLDS[i]] = metrics_dict
                    ap_calculator.reset()
                cur_criterion = "mAP_0.5"
                cur_best = reduce_tensor(_metrics_dict[0.5]['mAP']).item() if self.val_dist else _metrics_dict[0.5]['mAP']
                # cur_criterion = "sem_acc"
            else:
                cur_criterion = "iou_rate_0.5"
                cur_best = np.mean(self.log[phase][cur_criterion])
            if cur_best > self.best[cur_criterion]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))
                self._log("current train_loss: {}".format(np.mean(self.log["train"]["loss"])))
                self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
                self.best["epoch"] = epoch_id + 1
                self.best["loss"] = np.mean(self.log[phase]["loss"])
                self.best["ref_loss"] = np.mean(self.log[phase]["ref_loss"])
                self.best["lang_loss"] = np.mean(self.log[phase]["lang_loss"])
                self.best["objectness_loss"] = np.mean(self.log[phase]["objectness_loss"])
                self.best["vote_loss"] = np.mean(self.log[phase]["vote_loss"])
                self.best["box_loss"] = np.mean(self.log[phase]["box_loss"])
                self.best["relation_loss"] = np.mean(self.log[phase]["relation_loss"])
                self.best["color_loss"] = np.mean(self.log[phase]["color_loss"])
                self.best["shape_loss"] = np.mean(self.log[phase]["shape_loss"])
                self.best["size_loss"] = np.mean(self.log[phase]["size_loss"])
                self.best["lang_acc"] = np.mean(self.log[phase]["lang_acc"])
                self.best["ref_acc"] = np.mean(self.log[phase]["ref_acc"])
                self.best["obj_acc"] = np.mean(self.log[phase]["obj_acc"])
                self.best["pos_ratio"] = np.mean(self.log[phase]["pos_ratio"])
                self.best["neg_ratio"] = np.mean(self.log[phase]["neg_ratio"])
                self.best["iou_rate_0.25"] = np.mean(self.log[phase]["iou_rate_0.25"])
                self.best["iou_rate_0.5"] = np.mean(self.log[phase]["iou_rate_0.5"])
                self.best["sem_acc"] = np.mean(self.log[phase]["sem_acc"])
                self.best["relation_acc"] = np.mean(self.log[phase]["relation_acc"])
                self.best["color_acc"] = np.mean(self.log[phase]["color_acc"])
                self.best["shape_acc"] = np.mean(self.log[phase]["shape_acc"])
                self.best["size_acc"] = np.mean(self.log[phase]["size_acc"])
                if not self.reference:
                    self.best["mAP_0.25"] = reduce_tensor(_metrics_dict[0.25]['mAP']).item() if self.val_dist else _metrics_dict[0.25]['mAP']
                    self.best["mAP_0.5"] = reduce_tensor(_metrics_dict[0.5]['mAP']).item() if self.val_dist else _metrics_dict[0.5]['mAP']

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                # torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))
                if self.local_rank == 0:
                    torch.save(self.model.module.state_dict(), os.path.join(model_root, "model.pth"))
                    torch.save(self.model.state_dict(), os.path.join(model_root, "model_no_module.pth"))
                elif self.local_rank == -1:
                    torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

                # save check point
                self._log("saving checkpoint...\n")
                if self.local_rank in [0, -1]:
                    save_dict = {
                        "epoch": epoch_id,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                    }
                    checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                    torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

    def _dump_log(self, phase):
        log = {
            "loss": ["loss", "ref_loss", "lang_loss", "objectness_loss", "vote_loss", "box_loss", "relation_loss", "color_loss", "shape_loss", "size_loss"],
            "score": ["lang_acc", "ref_acc", "obj_acc", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5", "sem_acc", "relation_acc", "color_acc", "shape_acc", "size_acc"]
        }
        for key in log:
            for item in log[key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    self._global_iter_id
                )

    def _finish(self, epoch_id):
        if self.local_rank in [0, -1]:
            # print best
            self._best_report()
            # save check point
            self._log("saving checkpoint...\n")
            if self.local_rank in [0, -1]:
                save_dict = {
                    "epoch": epoch_id,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                }
                checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint_last.tar"))

            # save model
            self._log("saving last models...\n")
            model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
            if self.local_rank == 0:
                torch.save(self.model.module.state_dict(), os.path.join(model_root, "model_last.pth"))
            else:
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

            # export
            for phase in ["train", "val"]:
                self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))


        # # export
        # for phase in ["train", "val"]:
        #     self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * np.ceil(self._total_iter["train"] / self.val_step) * mean_est_val_time
        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_relation_loss=round(np.mean([v for v in self.log["train"]["relation_loss"]]), 5),
            train_color_loss=round(np.mean([v for v in self.log["train"]["color_loss"]]), 5),
            train_shape_loss=round(np.mean([v for v in self.log["train"]["shape_loss"]]), 5),
            train_size_loss=round(np.mean([v for v in self.log["train"]["size_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_sem_acc=round(np.mean([v for v in self.log["train"]["sem_acc"]]), 5),
            train_relation_acc=round(np.mean([v for v in self.log["train"]["relation_acc"]]), 5),
            train_color_acc=round(np.mean([v for v in self.log["train"]["color_acc"]]), 5),
            train_shape_acc=round(np.mean([v for v in self.log["train"]["shape_acc"]]), 5),
            train_size_acc=round(np.mean([v for v in self.log["train"]["size_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
            train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
            train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
            train_relation_loss=round(np.mean([v for v in self.log["train"]["relation_loss"]]), 5),
            train_color_loss=round(np.mean([v for v in self.log["train"]["color_loss"]]), 5),
            train_shape_loss=round(np.mean([v for v in self.log["train"]["shape_loss"]]), 5),
            train_size_loss=round(np.mean([v for v in self.log["train"]["size_loss"]]), 5),
            train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
            train_sem_acc=round(np.mean([v for v in self.log["train"]["sem_acc"]]), 5),
            train_relation_acc=round(np.mean([v for v in self.log["train"]["relation_acc"]]), 5),
            train_color_acc=round(np.mean([v for v in self.log["train"]["color_acc"]]), 5),
            train_shape_acc=round(np.mean([v for v in self.log["train"]["shape_acc"]]), 5),
            train_size_acc=round(np.mean([v for v in self.log["train"]["size_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
            train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
            train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            val_ref_loss=round(np.mean([v for v in self.log["val"]["ref_loss"]]), 5),
            val_lang_loss=round(np.mean([v for v in self.log["val"]["lang_loss"]]), 5),
            val_objectness_loss=round(np.mean([v for v in self.log["val"]["objectness_loss"]]), 5),
            val_vote_loss=round(np.mean([v for v in self.log["val"]["vote_loss"]]), 5),
            val_box_loss=round(np.mean([v for v in self.log["val"]["box_loss"]]), 5),
            val_relation_loss=round(np.mean([v for v in self.log["val"]["relation_loss"]]), 5),
            val_color_loss=round(np.mean([v for v in self.log["val"]["color_loss"]]), 5),
            val_shape_loss=round(np.mean([v for v in self.log["val"]["shape_loss"]]), 5),
            val_size_loss=round(np.mean([v for v in self.log["val"]["size_loss"]]), 5),
            val_lang_acc=round(np.mean([v for v in self.log["val"]["lang_acc"]]), 5),
            val_ref_acc=round(np.mean([v for v in self.log["val"]["ref_acc"]]), 5),
            val_obj_acc=round(np.mean([v for v in self.log["val"]["obj_acc"]]), 5),
            val_sem_acc=round(np.mean([v for v in self.log["val"]["sem_acc"]]), 5),
            val_relation_acc=round(np.mean([v for v in self.log["val"]["relation_acc"]]), 5),
            val_color_acc=round(np.mean([v for v in self.log["val"]["color_acc"]]), 5),
            val_shape_acc=round(np.mean([v for v in self.log["val"]["shape_acc"]]), 5),
            val_size_acc=round(np.mean([v for v in self.log["val"]["size_acc"]]), 5),
            val_pos_ratio=round(np.mean([v for v in self.log["val"]["pos_ratio"]]), 5),
            val_neg_ratio=round(np.mean([v for v in self.log["val"]["neg_ratio"]]), 5),
            val_iou_rate_25=round(np.mean([v for v in self.log["val"]["iou_rate_0.25"]]), 5),
            val_iou_rate_5=round(np.mean([v for v in self.log["val"]["iou_rate_0.5"]]), 5),
        )
        self._log(epoch_report)
    
    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            ref_loss=round(self.best["ref_loss"], 5),
            lang_loss=round(self.best["lang_loss"], 5),
            objectness_loss=round(self.best["objectness_loss"], 5),
            vote_loss=round(self.best["vote_loss"], 5),
            box_loss=round(self.best["box_loss"], 5),
            relation_loss=round(self.best["relation_loss"], 5),
            color_loss=round(self.best["color_loss"], 5),
            shape_loss=round(self.best["shape_loss"], 5),
            size_loss=round(self.best["size_loss"], 5),
            lang_acc=round(self.best["lang_acc"], 5),
            ref_acc=round(self.best["ref_acc"], 5),
            obj_acc=round(self.best["obj_acc"], 5),
            sem_acc=round(self.best["sem_acc"], 5),
            relation_acc=round(self.best["relation_acc"], 5),
            color_acc=round(self.best["color_acc"], 5),
            shape_acc=round(self.best["shape_acc"], 5),
            size_acc=round(self.best["size_acc"], 5),
            map_25=round(self.best["mAP_0.25"], 5),
            map_5=round(self.best["mAP_0.5"], 5),
            pos_ratio=round(self.best["pos_ratio"], 5),
            neg_ratio=round(self.best["neg_ratio"], 5),
            iou_rate_25=round(self.best["iou_rate_0.25"], 5),
            iou_rate_5=round(self.best["iou_rate_0.5"], 5),
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
