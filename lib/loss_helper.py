# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness

def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = data_dict['seed_xyz'].shape[0]
    num_seed = data_dict['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = data_dict['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = data_dict['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += data_dict['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict['aggregated_vote_xyz']
    gt_center = data_dict['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = data_dict['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict['center']
    gt_center = data_dict['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict['box_label_mask']
    objectness_label = data_dict['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(data_dict['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(data_dict['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(data_dict['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(data_dict['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(data_dict['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(data_dict['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(data_dict['size_residuals_normalized'].contiguous()*size_label_one_hot_tiled.contiguous(), 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(data_dict['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def compute_reference_loss(data_dict, config):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """

    # unpack
    cluster_preds = data_dict["cluster_ref"] # (B, num_proposal)

    # predicted bbox
    pred_ref = data_dict['cluster_ref'].detach().cpu().numpy() # (B,)
    pred_center = data_dict['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # ground truth bbox
    gt_center = data_dict['ref_center_label'].cpu().numpy() # (B,3)
    gt_heading_class = data_dict['ref_heading_class_label'].cpu().numpy() # B
    gt_heading_residual = data_dict['ref_heading_residual_label'].cpu().numpy() # B
    gt_size_class = data_dict['ref_size_class_label'].cpu().numpy() # B
    gt_size_residual = data_dict['ref_size_residual_label'].cpu().numpy() # B,3
    # convert gt bbox parameters to bbox corners
    gt_obb_batch = config.param2obb_batch(gt_center[:, 0:3], gt_center, gt_heading_residual,
                    gt_size_class, gt_size_residual)
    gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

    # compute the iou score for all predictd positive ref
    batch_size, num_proposals = cluster_preds.shape
    labels = np.zeros((batch_size, num_proposals))
    for i in range(pred_ref.shape[0]):
        # convert the bbox parameters to bbox corners
        pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_center[i, :, 0:3], pred_heading_residual[i],
                    pred_size_class[i], pred_size_residual[i])
        pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[i], (num_proposals, 1, 1)))
        labels[i, ious.argmax()] = 1 # treat the bbox with highest iou score as the gt

    cluster_labels = torch.FloatTensor(labels).cuda()

    # reference loss
    criterion = SoftmaxRankingLoss()
    loss = criterion(cluster_preds, cluster_labels.float().clone())

    return loss, cluster_preds, cluster_labels


def get_topk_proposals(data_dict, idx_, batch_id, config, k):
    if len(idx_) < k:
        k = len(idx_)

    # unpack
    cluster_preds = data_dict["cluster_ref"] # (B, num_proposal)

    # predicted bbox
    pred_center = data_dict['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # ground truth bbox
    gt_center = data_dict['ref_center_label'].cpu().numpy() # (B,3)
    gt_heading_class = data_dict['ref_heading_class_label'].cpu().numpy() # B
    gt_heading_residual = data_dict['ref_heading_residual_label'].cpu().numpy() # B
    gt_size_class = data_dict['ref_size_class_label'].cpu().numpy() # B
    gt_size_residual = data_dict['ref_size_residual_label'].cpu().numpy() # B,3
    # convert gt bbox parameters to bbox corners
    gt_obb_batch = config.param2obb_batch(gt_center[:, 0:3], gt_center, gt_heading_residual,
                    gt_size_class, gt_size_residual)
    gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

    # compute the iou score for all predictd positive ref
    batch_size, num_proposals = cluster_preds[:, idx_].shape

    # print("pred_heading_class[batch_id, idx_]", pred_heading_class[batch_id, idx_].shape)
    # print("len(idx_)", len(idx_))
    pred_obb_batch = config.param2obb_batch(pred_center[batch_id, idx_, 0:3], pred_center[batch_id, idx_, 0:3], pred_heading_residual[batch_id, idx_],
                    pred_size_class[batch_id, idx_], pred_size_residual[batch_id, idx_])
    pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
    ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[batch_id], (num_proposals, 1, 1)))

    return torch.topk(torch.tensor(ious), k, dim=-1)[1]

def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])

    return loss

def compute_relation_prediction_loss(data_dict, fully_sup, config):
    rel_preds = data_dict["rel_pred"] # length: total number of relations in a batch
    loss = torch.zeros(1)[0].cuda()
    relation_criterion_binary = nn.BCELoss(reduction='none').cuda()
    gt_mapping = {0: [1, 0, 1, 0, 0, 0, 0, 0.5, 0], 
                1: [0, 1, 1, 0, 0, 0, 0, 0.5, 0], 
                2: [0.5, 0.5, 1, 0.5, 0, 0.5, 0, 0.5, 0],
                3: [0, 0, 0.5, 1, 0, 0, 0, 0.5, 0],
                4: [0, 0, 0, 0, 1, 0, 0, 0, 0],
                5: [0, 0, 0.5, 0, 0, 1, 0, 0, 0],
                6: [0, 0, 0, 0, 0, 0, 1, 0, 0],
                7: [0.5, 0.5, 0.5, 0.5, 0, 0, 0, 1, 0],
                8: [0, 0, 0, 0, 0, 0, 0, 0, 1]} # add for distribution
    if not fully_sup:
        if len(rel_preds) != 0: 
            count = 0
            for rel_pred in rel_preds:
                sub_f, obj_f, gt, pred = rel_pred['sub_f'], rel_pred['obj_f'], rel_pred['gt'], rel_pred['pred']
                sub_conf, obj_conf = rel_pred['sub_conf'], rel_pred['obj_conf'] # (Nsub), (Nobj)
                if pred.shape[0] == 0 or pred.shape[1] == 0:
                    continue
                Nsub, Nobj, NumRel = pred.shape
                pred = pred.view((Nsub * Nobj, NumRel))

                # binary ce
                gt_binary = torch.tensor(gt_mapping[int(gt)]).float().cuda() 
                gt_binary = torch.repeat_interleave(gt_binary.unsqueeze(0), repeats=Nsub * Nobj, dim=0) # (Nsub * Nobj) * NumRel
                rel_loss = torch.mean(relation_criterion_binary(torch.sigmoid(pred), gt_binary), dim=-1) # (Nsub * Nobj)

                # rel_loss = rel_loss.view((Nsub, Nobj))
                rel_loss = rel_loss.min(dim=0)[0]

                loss += rel_loss
                # assert rel_loss > 0
                count += 1
            if count > 0:
                loss = loss/count
    else:
        k=1
        if len(rel_preds) != 0: 
            count = 0
            for rel_pred in rel_preds:
                sub_f, obj_f, gt, pred = rel_pred['sub_f'], rel_pred['obj_f'], rel_pred['gt'], rel_pred['pred']
                sub_xyz, obj_xyz, rel_ref_gt_xyz = rel_pred['sub_xyz'], rel_pred['obj_xyz'], rel_pred['rel_ref_gt_xyz']
                sub_conf, obj_conf = rel_pred['sub_conf'], rel_pred['obj_conf'] # (Nsub), (Nobj)
                sub_idx_, obj_idx_, batch_id = rel_pred['sub_idx_'], rel_pred['obj_idx_'], rel_pred['batch_id']
                assert sub_conf.shape[0] == pred.shape[0]
                assert obj_conf.shape[0] == pred.shape[1]
                if pred.shape[0] == 0 or pred.shape[1] == 0:
                    continue
                Nsub, Nobj, NumRel = pred.shape
                pred = pred.view((Nsub * Nobj, NumRel))

                # binary ce
                gt_binary = torch.tensor(gt_mapping[int(gt)]).float().cuda() 
                gt_binary = torch.repeat_interleave(gt_binary.unsqueeze(0), repeats=Nsub * Nobj, dim=0) # (Nsub * Nobj) * NumRel
                rel_loss = torch.mean(relation_criterion_binary(torch.sigmoid(pred), gt_binary), dim=-1) # (Nsub * Nobj)
                rel_loss = rel_loss.view((Nsub, Nobj))

                # assert (sub_f and obj_f) == False
                if sub_f:     
                    # use iou
                    if len(sub_idx_) < k:
                        rel_loss = rel_loss.min(dim=1)[0].mean()
                    else:
                        chosen_idx = get_topk_proposals(data_dict, sub_idx_.cpu(), batch_id, config, k)
                        rel_loss = rel_loss.min(dim=1)[0][chosen_idx].mean()

                elif obj_f:                 
                    # use iou
                    if len(obj_idx_) < k:
                        rel_loss = rel_loss.min(dim=0)[0].mean()
                    else:
                        chosen_idx = get_topk_proposals(data_dict, obj_idx_.cpu(), batch_id, config, k)
                        rel_loss = rel_loss.min(dim=0)[0][chosen_idx].mean()
                else:
                    rel_loss = rel_loss.min()

                loss += rel_loss
                count += 1
            if count > 0:
                loss = loss/count

    return loss

def compute_color_prediction_loss(data_dict, fully_sup, config):
    col_preds = data_dict["col_pred"] # length: total number of relations in a batch
    loss = torch.zeros(1)[0].cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    if not fully_sup:
        if len(col_preds) != 0:
            count = 0
            for col_pred in col_preds:
                pred_col, gt_col = col_pred['pred_col'], col_pred['gt_col']
                col_conf = col_pred['col_conf']
                gt_col = torch.repeat_interleave(gt_col.unsqueeze(0), repeats=pred_col.shape[0], dim=0) # Ncol * NumCol
                col_loss = criterion(pred_col, gt_col.long().cuda())

                sel_min = col_loss.min(dim=0)[1]

                col_loss = col_loss.min(dim=0)[0]
                loss += col_loss
                count += 1

            if count > 0:
                loss = loss/count
    else:
        k=1
        if len(col_preds) != 0:
            count = 0
            for col_pred in col_preds:
                pred_col, gt_col = col_pred['pred_col'], col_pred['gt_col'] # (Ncol, 11), (1)
                col_xyz, col_ref_gt_xyz = col_pred['col_xyz'], col_pred['col_ref_gt_xyz'] # (Ncol, 3), (1,3)
                col_conf = col_pred['col_conf']
                col_idx_, batch_id = col_pred['col_idx_'], col_pred['batch_id']

                if len(col_idx_) < k:
                    gt_col = torch.repeat_interleave(gt_col.unsqueeze(0), repeats=pred_col.shape[0], dim=0) # Ncol * NumCol
                    col_loss = criterion(pred_col, gt_col.long().cuda()).mean()
                else:
                    chosen_idx = get_topk_proposals(data_dict, col_idx_.cpu(), batch_id, config, k)
                    pred_col = pred_col[chosen_idx] # (k, 11)
                    gt_col = torch.repeat_interleave(gt_col.unsqueeze(0), repeats=pred_col.shape[0], dim=0) # Ncol * NumCol
                    col_loss = criterion(pred_col, gt_col.long().cuda()).mean()
                
                loss += col_loss
                count += 1

            if count > 0:
                loss = loss/count
    
    return loss

def compute_shape_prediction_loss(data_dict, fully_sup, config):
    shape_preds = data_dict["shape_pred"] # length: total number of relations in a batch
    loss = torch.zeros(1)[0].cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    if not fully_sup:
        if len(shape_preds) != 0:
            count = 0
            for shape_pred in shape_preds:
                pred_shape, gt_shape = shape_pred['pred_shape'], shape_pred['gt_shape']
                shape_conf = shape_pred['shape_conf']
                gt_shape = torch.repeat_interleave(gt_shape.unsqueeze(0), repeats=pred_shape.shape[0], dim=0) # Nshape * Numshape
                shape_loss = criterion(pred_shape, gt_shape.long().cuda())

                sel_min = shape_loss.min(dim=0)[1]

                shape_loss = shape_loss.min(dim=0)[0]
                loss += shape_loss
                count += 1

            if count > 0:
                loss = loss/count
    else:
        k=1
        if len(shape_preds) != 0:
            count = 0
            for shape_pred in shape_preds:
                pred_shape, gt_shape = shape_pred['pred_shape'], shape_pred['gt_shape'] # (Nshape, 11), (1)
                shape_xyz, shape_ref_gt_xyz = shape_pred['shape_xyz'], shape_pred['shape_ref_gt_xyz'] # (Nshape, 3), (1,3)
                shape_conf = shape_pred['shape_conf']
                shape_idx_, batch_id = shape_pred['shape_idx_'], shape_pred['batch_id']

                if len(shape_idx_) < k:
                    gt_shape = torch.repeat_interleave(gt_shape.unsqueeze(0), repeats=pred_shape.shape[0], dim=0) # Nshape * Numshape
                    shape_loss = criterion(pred_shape, gt_shape.long().cuda()).mean()
                else:
                    chosen_idx = get_topk_proposals(data_dict, shape_idx_.cpu(), batch_id, config, k)
                    pred_shape = pred_shape[chosen_idx] # (k, 11)
                    gt_shape = torch.repeat_interleave(gt_shape.unsqueeze(0), repeats=pred_shape.shape[0], dim=0) # Nshape * Numshape
                    shape_loss = criterion(pred_shape, gt_shape.long().cuda()).mean()
                
                loss += shape_loss
                count += 1

            if count > 0:
                loss = loss/count
    
    return loss

def compute_size_prediction_loss(data_dict, fully_sup, config):
    size_preds = data_dict["size_pred"] # length: total number of relations in a batch
    loss = torch.zeros(1)[0].cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')
    if not fully_sup:
        if len(size_preds) != 0:
            count = 0
            for size_pred in size_preds:
                pred_size, gt_size = size_pred['pred_size'], size_pred['gt_size']
                size_conf = size_pred['size_conf']
                gt_size = torch.repeat_interleave(gt_size.unsqueeze(0), repeats=pred_size.shape[0], dim=0) # Nsize * Numsize
                size_loss = criterion(pred_size, gt_size.long().cuda())

                sel_min = size_loss.min(dim=0)[1]

                size_loss = size_loss.min(dim=0)[0]
                loss += size_loss
                count += 1

            if count > 0:
                loss = loss/count
    else:
        k=1
        if len(size_preds) != 0:
            count = 0
            for size_pred in size_preds:
                pred_size, gt_size = size_pred['pred_size'], size_pred['gt_size'] # (Nsize, 11), (1)
                size_xyz, size_ref_gt_xyz = size_pred['size_xyz'], size_pred['size_ref_gt_xyz'] # (Nsize, 3), (1,3)
                size_conf = size_pred['size_conf']
                size_idx_, batch_id = size_pred['size_idx_'], size_pred['batch_id']

                if len(size_idx_) < k:
                    gt_size = torch.repeat_interleave(gt_size.unsqueeze(0), repeats=pred_size.shape[0], dim=0) # Nsize * Numsize
                    size_loss = criterion(pred_size, gt_size.long().cuda()).mean()
                else:
                    chosen_idx = get_topk_proposals(data_dict, size_idx_.cpu(), batch_id, config, k)
                    pred_size = pred_size[chosen_idx] # (k, 11)
                    gt_size = torch.repeat_interleave(gt_size.unsqueeze(0), repeats=pred_size.shape[0], dim=0) # Nsize * Numsize
                    size_loss = criterion(pred_size, gt_size.long().cuda()).mean()
                
                loss += size_loss
                count += 1

            if count > 0:
                loss = loss/count
    
    return loss

def get_loss(data_dict, config, detection=True, reference=True, use_lang_classifier=False, 
            relation_prediction=False, color_prediction=False, shape_prediction=False, size_prediction=False,
            fully_sup=False, epoch_id=0, prepare_epoch=0):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    pretask = relation_prediction or color_prediction or shape_prediction or size_prediction

    # Vote loss
    vote_loss = compute_vote_loss(data_dict)

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    num_proposal = objectness_label.shape[1]
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    data_dict['objectness_label'] = objectness_label
    data_dict['objectness_mask'] = objectness_mask
    data_dict['object_assignment'] = object_assignment
    data_dict['pos_ratio'] = torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    data_dict['neg_ratio'] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss

    if detection:
        data_dict['vote_loss'] = vote_loss
        data_dict['objectness_loss'] = objectness_loss
        data_dict['center_loss'] = center_loss
        data_dict['heading_cls_loss'] = heading_cls_loss
        data_dict['heading_reg_loss'] = heading_reg_loss
        data_dict['size_cls_loss'] = size_cls_loss
        data_dict['size_reg_loss'] = size_reg_loss
        data_dict['sem_cls_loss'] = sem_cls_loss
        data_dict['box_loss'] = box_loss
    else:
        data_dict['vote_loss'] = torch.zeros(1)[0].cuda()
        data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
        data_dict['center_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['heading_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['size_reg_loss'] = torch.zeros(1)[0].cuda()
        data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['box_loss'] = torch.zeros(1)[0].cuda()

    if reference:
        # Reference loss
        ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        data_dict["cluster_labels"] = cluster_labels
        data_dict["ref_loss"] = ref_loss
    elif pretask:
        # Reference loss
        ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
        data_dict["ref_loss"] = 0*ref_loss # support for distribution
        data_dict["cluster_labels"] = data_dict['objectness_label'].new_zeros(data_dict['objectness_label'].shape).cuda()
        data_dict["cluster_ref"] = data_dict['objectness_label'].new_zeros(data_dict['objectness_label'].shape).float().cuda()
    else:
        data_dict["cluster_labels"] = data_dict['objectness_label'].new_zeros(data_dict['objectness_label'].shape).cuda()
        data_dict["cluster_ref"] = data_dict['objectness_label'].new_zeros(data_dict['objectness_label'].shape).float().cuda()

        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()

    if use_lang_classifier and (reference or pretask):
        data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    else:
        data_dict["lang_loss"] = torch.zeros(1)[0].cuda()
    
    if relation_prediction and epoch_id >= prepare_epoch:
        data_dict['relation_loss'] = compute_relation_prediction_loss(data_dict, fully_sup, config)
    else:
        data_dict['relation_loss'] = torch.zeros(1)[0].cuda()
    
    if color_prediction and epoch_id >= prepare_epoch:
        data_dict['color_loss'] = compute_color_prediction_loss(data_dict, fully_sup, config)
    else:
        data_dict['color_loss'] = torch.zeros(1)[0].cuda()

    if shape_prediction and epoch_id >= prepare_epoch:
        data_dict['shape_loss'] = compute_shape_prediction_loss(data_dict, fully_sup, config)
    else:
        data_dict['shape_loss'] = torch.zeros(1)[0].cuda()

    if size_prediction and epoch_id >= prepare_epoch:
        data_dict['size_loss'] = compute_size_prediction_loss(data_dict, fully_sup, config)
    else:
        data_dict['size_loss'] = torch.zeros(1)[0].cuda()

    # Final loss function
    loss = data_dict['vote_loss'] + 0.5*data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1*data_dict['sem_cls_loss'] \
        + 0.1*data_dict["ref_loss"] + 0.1*data_dict["lang_loss"] \
        + 0.05*data_dict['relation_loss'] + 0.05*data_dict['color_loss'] + 0.05*data_dict['shape_loss'] + 0.05*data_dict['size_loss']
    
    loss *= 10 # amplify

    data_dict['loss'] = loss

    return loss, data_dict
