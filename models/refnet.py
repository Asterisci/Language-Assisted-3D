import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule
from models.match_module import MatchModule

from models.MLCVNet.voting_module import VotingModule as MLCVNetVotingModule
from models.MLCVNet.proposal_module import ProposalModule as MLCVNetProposalModule

from utils.nn_distance import nn_distance

class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
    use_lang_classifier=True, use_bidir=False, no_reference=False, 
    relation_prediction=False, color_prediction=False, shape_prediction=False, size_prediction=False,
    emb_size=300, hidden_size=256, prepare_epoch=0, MLCVNet_backbone=False):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir      
        self.no_reference = no_reference
        self.relation_prediction = relation_prediction
        self.color_prediction = color_prediction
        self.shape_prediction = shape_prediction
        self.size_prediction = size_prediction
        self.prepare_epoch = prepare_epoch
        self.pretask = relation_prediction or color_prediction or shape_prediction or size_prediction
        self.MLCVNet_backbone = MLCVNet_backbone


        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning

        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        if MLCVNet_backbone:
            # Hough voting
            self.vgen = MLCVNetVotingModule(self.vote_factor, 256)

            # Vote aggregation and object proposal
            self.proposal = MLCVNetProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
        else: 
            # Hough voting
            self.vgen = VotingModule(self.vote_factor, 256)

            # Vote aggregation and object proposal
            self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
        if not no_reference or self.pretask:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size)

        if self.relation_prediction:
            # --------- RELATION PREDICTION ----------
            self.relation_merge = nn.Sequential(
                    nn.Linear(2 * (128+num_proposal), (128+num_proposal)),
                    nn.ReLU(inplace=True))
            self.vis2rel = nn.Linear((128+num_proposal), 9)
        if self.color_prediction:
            # --------- COLOR PREDICTION ----------
            self.vis2col = nn.Linear((128+num_proposal), 12)
        if self.shape_prediction:
            # --------- SHAPE PREDICTION ----------
            self.vis2shape = nn.Linear((128+num_proposal), 8)
        if self.size_prediction:
            # --------- SIZE PREDICTION ----------
            self.vis2size = nn.Linear((128+num_proposal), 9)

    def forward(self, data_dict, epoch_id):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)

        if not self.no_reference or self.pretask:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.match(data_dict)
        
        if self.pretask and (epoch_id>=self.prepare_epoch):
            sem_cls_scores = data_dict['sem_cls_scores'] # (batch_size, num_proposal, 18)
            objectness_scores = data_dict['objectness_scores'] # (batch_size, num_proposal, 2)
            semantic_preds = sem_cls_scores.max(-1)[1]    # (batch_size, num_proposal), long
            objectness_preds = objectness_scores.max(-1)[1]    # (batch_size, num_proposal), long
            aggregated_vote_xyz = data_dict['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)

            # match vote_features and language
            fuse_features = data_dict['aggregated_vote_features'] # batch_size, num_proposal, 128
            # unpack outputs from language branch
            lang_feat = data_dict["lang_emb"] # batch_size, lang_size
            lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposal, 1) # batch_size, num_proposals, lang_size
            # fuse
            fuse_features = torch.cat([fuse_features, lang_feat], dim=-1) # batch_size, num_proposals, 128 + lang_size

            aggregated_vote_features = fuse_features # batch_size, num_proposals, 128 + lang_size
            assert sem_cls_scores.shape[1] == aggregated_vote_features.shape[1]
            assert objectness_scores.shape[1] == aggregated_vote_features.shape[1]
            ref_center_label = data_dict["ref_center_label"].unsqueeze(1) # (batch_size, 1, 3)
            confidences = data_dict["cluster_ref"] # (batch_size, num_proposal)
        

        if self.relation_prediction and (epoch_id>=self.prepare_epoch):
            #######################################
            #                                     #
            #      RELATION PREDICTION BRANCH     #
            #                                     #
            #######################################
            relmat = data_dict["relmat"] # (B, 3, 5)
            # print(relmat)
            num_rel = data_dict["num_rel"] # (B, 1)
            rel_preds = [] # length: total relation numbers in a batch
            assert relmat.shape[0] == data_dict["unique_multiple"].shape[0]
            correct = torch.zeros(1).cuda()
            wrong = torch.zeros(1).cuda()
            for i in range(relmat.shape[0]):
                # if data_dict["unique_multiple"][i]==0: # first train unique scene for 10 epoch, then multiple scene
                #     continue
                for j in range(int(num_rel[i])):
                    sub_idx = (semantic_preds[i] == relmat[i][j][0]) * (objectness_preds[i] == 1) # only object boxes are considered
                    sub_idx = torch.nonzero(sub_idx, as_tuple=False).view(-1) # (num_proposal)
                    obj_idx = (semantic_preds[i] == relmat[i][j][1]) * (objectness_preds[i] == 1) # only object boxes are considered
                    obj_idx = torch.nonzero(obj_idx, as_tuple=False).view(-1) # (num_proposal)
                    Nsub = len(sub_idx)
                    Nobj = len(obj_idx)
                    # print("found", Nsub, relmat[i][j][0].cpu(), "(subjects)   and", Nobj, relmat[i][j][1].cpu(),  "(objects)")
                    if (Nsub>0) and (Nobj>0):
                        # suc_case += 1
                        sub_f = False
                        obj_f = False
                        if relmat[i][j][3] != -1:
                            sub_f = True
                        if relmat[i][j][4] != -1:
                            obj_f = True
                        sub_feat = aggregated_vote_features[i][sub_idx.long()] # (Nsub, 128)
                        obj_feat = aggregated_vote_features[i][obj_idx.long()] # (Nobj, 128)
                        sub_xyz = aggregated_vote_xyz[i][sub_idx.long()] # (Nsub, 3)
                        obj_xyz = aggregated_vote_xyz[i][obj_idx.long()] # (Nobj, 3)
                        sub_conf = confidences[i][sub_idx.long()] # (Nsub)
                        obj_conf = confidences[i][obj_idx.long()] # (Nobj)
                        subject = torch.repeat_interleave(sub_feat.unsqueeze(dim=1), repeats=Nobj, dim=1)
                        object = torch.repeat_interleave(obj_feat.unsqueeze(dim=0), repeats=Nsub, dim=0)
                        fusion = torch.cat([subject, object], dim=-1).view(Nsub * Nobj, 2*(128+self.num_proposal))
                        pred = self.vis2rel(self.relation_merge(fusion)).view(Nsub, Nobj, 9) # big matrix!!! (Nsub, Nobj, NumRelations)
                        assert len(pred.shape) == 3
                        rel_preds.append({'sub': relmat[i][j][0], 'obj': relmat[i][j][1], 'pred': pred, 'gt': relmat[i][j][2],
                                            'sub_f': sub_f, 'obj_f': obj_f, 'sub_xyz': sub_xyz, 'obj_xyz': obj_xyz, 'rel_ref_gt_xyz': ref_center_label[i],
                                            'sub_conf': sub_conf, 'obj_conf': obj_conf,
                                            'sub_idx_': sub_idx.long(), 'obj_idx_': obj_idx.long(), 'batch_id': i})

                        test_pred = (pred.view(-1, 9).mean(0) > 0).long() # (8)
                        gt_mapping = {0: [1, 0, 1, 0, 0, 0, 0, 0.5, 0], 
                                        1: [0, 1, 1, 0, 0, 0, 0, 0.5, 0], 
                                        2: [0.5, 0.5, 1, 0.5, 0, 0.5, 0, 0.5, 0],
                                        3: [0, 0, 0.5, 1, 0, 0, 0, 0.5, 0],
                                        4: [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                        5: [0, 0, 0.5, 0, 0, 1, 0, 0, 0],
                                        6: [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                        7: [0.5, 0.5, 0.5, 0.5, 0, 0, 0, 1, 0],
                                        8: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
                        gt_binary = torch.tensor(gt_mapping[int(relmat[i][j][2])]).float().cuda() 
                        correct += ((test_pred-gt_binary)==0).sum()
                        wrong += (((test_pred-gt_binary)==1).sum() + ((test_pred-gt_binary)==-1).sum())
            data_dict["rel_pred"] = rel_preds
            data_dict['rel_correct'] = correct
            data_dict['rel_wrong'] = wrong
        
        if self.color_prediction and (epoch_id>=self.prepare_epoch):
            colmat = data_dict["colmat"] # (B, 1, 2)
            # print(relmat)
            num_col = data_dict["num_col"] # (B, 1)
            col_preds = []
            assert colmat.shape[0] == data_dict["unique_multiple"].shape[0]
            correct = torch.zeros(1).cuda()
            wrong = torch.zeros(1).cuda()
            for i in range(sem_cls_scores.shape[0]):
                # if data_dict["unique_multiple"][i]==0: # first train unique scene for 10 epoch, then multiple scene
                #     continue
                for j in range(int(num_col[i])):
                    col_idx = (semantic_preds[i] == data_dict["object_cat"][i]) * (objectness_preds[i] == 1) # only object boxes are considered
                    col_idx = torch.nonzero(col_idx, as_tuple=False).view(-1) # (num_proposal)
                    Ncol = len(col_idx)
                    if Ncol>0:
                        col_feat = aggregated_vote_features[i][col_idx.long()] # (Ncol, 128)
                        col_xyz = aggregated_vote_xyz[i][col_idx.long()] # (Ncol, 3)
                        col_conf = confidences[i][col_idx.long()] # (Ncol)
                        pred_col = self.vis2col(col_feat) # (Ncol, 11)
                        col_preds.append({'pred_col': pred_col, 'gt_col': colmat[i][j][1], 'col_xyz': col_xyz, 'col_ref_gt_xyz': ref_center_label[i], 
                                        'col_conf': col_conf, 'col_idx_': col_idx.long(), 'batch_id': i})
                        if torch.argmax(pred_col.mean(0), dim=-1).long() == colmat[i][j][1]:
                            correct += 1 
                        else:
                            wrong += 1
            data_dict["col_pred"] = col_preds
            data_dict['col_correct'] = correct
            data_dict['col_wrong'] = wrong

        if self.shape_prediction and (epoch_id>=self.prepare_epoch):
            shapemat = data_dict["shapemat"] # (B, 1, 2)
            # print(relmat)
            num_shape = data_dict["num_shape"] # (B, 1)
            shape_preds = []
            assert shapemat.shape[0] == data_dict["unique_multiple"].shape[0]
            correct = torch.zeros(1).cuda()
            wrong = torch.zeros(1).cuda()
            for i in range(sem_cls_scores.shape[0]):
                # if data_dict["unique_multiple"][i]==0: # first train unique scene for 10 epoch, then multiple scene
                #     continue
                for j in range(int(num_shape[i])):
                    shape_idx = (semantic_preds[i] == data_dict["object_cat"][i]) * (objectness_preds[i] == 1) # only object boxes are considered
                    shape_idx = torch.nonzero(shape_idx, as_tuple=False).view(-1) # (num_proposal)
                    Nshape = len(shape_idx)
                    # print("found", Nshape, "shapeored objects")
                    if Nshape>0:
                        shape_feat = aggregated_vote_features[i][shape_idx.long()] # (Nshape, 128)
                        shape_xyz = aggregated_vote_xyz[i][shape_idx.long()] # (Nshape, 3)
                        shape_conf = confidences[i][shape_idx.long()] # (Nshape)
                        pred_shape = self.vis2shape(shape_feat) # (Nshape, 11)
                        shape_preds.append({'pred_shape': pred_shape, 'gt_shape': shapemat[i][j][1], 'shape_xyz': shape_xyz, 'shape_ref_gt_xyz': ref_center_label[i], 
                                        'shape_conf': shape_conf, 'shape_idx_': shape_idx.long(), 'batch_id': i})
                        if torch.argmax(pred_shape.mean(0), dim=-1).long() == shapemat[i][j][1]:
                            correct += 1 
                        else:
                            wrong += 1
            data_dict["shape_pred"] = shape_preds
            data_dict['shape_correct'] = correct
            data_dict['shape_wrong'] = wrong

        if self.size_prediction and (epoch_id>=self.prepare_epoch):
            sizemat = data_dict["sizemat"] # (B, 1, 2)
            num_size = data_dict["num_size"] # (B, 1)
            size_preds = []
            assert sizemat.shape[0] == data_dict["unique_multiple"].shape[0]
            correct = torch.zeros(1).cuda()
            wrong = torch.zeros(1).cuda()
            for i in range(sem_cls_scores.shape[0]):
                # if data_dict["unique_multiple"][i]==0: # first train unique scene for 10 epoch, then multiple scene
                #     continue
                for j in range(int(num_size[i])):
                    size_idx = (semantic_preds[i] == data_dict["object_cat"][i]) * (objectness_preds[i] == 1) # only object boxes are considered
                    size_idx = torch.nonzero(size_idx, as_tuple=False).view(-1) # (num_proposal)
                    Nsize = len(size_idx)
                    # print("found", Nsize, "sizeored objects")
                    if Nsize>0:
                        size_feat = aggregated_vote_features[i][size_idx.long()] # (Nsize, 128)
                        size_xyz = aggregated_vote_xyz[i][size_idx.long()] # (Nsize, 3)
                        size_conf = confidences[i][size_idx.long()] # (Nsize)
                        pred_size = self.vis2size(size_feat) # (Nsize, 11)
                        size_preds.append({'pred_size': pred_size, 'gt_size': sizemat[i][j][1], 'size_xyz': size_xyz, 'size_ref_gt_xyz': ref_center_label[i], 
                                        'size_conf': size_conf, 'size_idx_': size_idx.long(), 'batch_id': i})
                        if torch.argmax(pred_size.mean(0), dim=-1).long() == sizemat[i][j][1]:
                            correct += 1 
                        else:
                            wrong += 1
            data_dict["size_pred"] = size_preds
            data_dict['size_correct'] = correct
            data_dict['size_wrong'] = wrong

        return data_dict
