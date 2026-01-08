import math
import torch
from torch import nn
import torch.nn.functional as F
from network.PFE import AdaptiveWeighting, Pairwise_Feature_Enhancer
from modules.cbam import CBAM
from network.DSwin import AdaSwinTransformer, SwinTransformer_Teacher
from network.PGA import PatchGraphAggregator

supported_arch = ["swin-base", "swin-tiny", "swin-small"]

# Paths to pretrained Swin Transformer weights
PRETRAIN_LIST = {
    "swin-small": "./pretrained/swin/swin_small_patch4_window7_224_22kto1k_finetune.pth",
    "swin-tiny": "./pretrained/swin/swin_tiny_patch4_window7_224_22kto1k_finetune.pth",
    "swin-base": "./pretrained/swin/swin_base_patch4_window7_224_22kto1k.pth",
}


def pdist(vectors):
    # Calculate Euclidean distance matrix
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def create_global_branch(arch: str, cfg: dict, only_teacher_model: bool = False):
    # Builder function for Swin Transformer backbone (Teacher or Student)
    # print(f"DEBUG: create_global_branch received arch: {arch}")
    swin_configs = {
        "swin-tiny": {
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24]
        },
        "swin-small": {
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24]
        },
        "swin-base": {
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32]
        },

    }
    arch_key = arch
    config = swin_configs[arch_key]
    embed_dim = config["embed_dim"]
    depth = config["depths"]
    num_heads = config["num_heads"]
    # print(f"DEBUG: Config for {arch_key} has embed_dim: {embed_dim}")

    pretrained_path = PRETRAIN_LIST[arch_key]

    pruning_loc = cfg["pruning_loc"]
    keep_rate = cfg["keep_rate"]

    if only_teacher_model:
        # Initialize Teacher Model (standard Swin)
        teacher_model = SwinTransformer_Teacher(
            img_size=cfg["image_size"], num_classes=0, window_size=cfg["window_size"],
            embed_dim=embed_dim, depths=depth, num_heads=num_heads,
        )
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        teacher_model.load_state_dict(checkpoint, strict=False)
        return teacher_model
    else:
        # Initialize Student Model (Adaptive Swin with pruning)
        model = AdaSwinTransformer(
            img_size=cfg["image_size"], num_classes=0, window_size=cfg["window_size"],
            embed_dim=embed_dim, depths=depth, num_heads=num_heads,
            keep_rate=keep_rate, pruning_loc=pruning_loc
        )
        global_feature_dim = model.num_features
        if cfg["load_pretrained"]:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            model.load_state_dict(checkpoint, strict=False)
        num_patches = model.layers[-1].input_resolution[0] * model.layers[-1].input_resolution[1]
        return model, pruning_loc, global_feature_dim, num_patches, len(depth)


class DynamicFusionModule(nn.Module):
    # Module to compute adaptive weights (alpha, beta) for feature fusion
    def __init__(self, input_dim):
        super(DynamicFusionModule, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, global_features, local_features):
        combined_feats = torch.cat([global_features, local_features], dim=1)
        weights = self.fc(combined_feats)
        alpha, beta = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
        return alpha, beta


class IGCC(nn.Module):
    # Main model class integrating Global Branch and Local GNN Branch
    def __init__(self, model_cfg, local_cfg):
        super(IGCC, self).__init__()

        self.NoGNN = model_cfg['NoGNN']
        self.NoAFF = model_cfg['NoAFF']
        self.NoPFE = model_cfg['NoPFE']
        self.NoMAPS = model_cfg['NoMAPS']
        self.arch = model_cfg["arch"]
        self.keep_rate = model_cfg["keep_rate"]
        # Initialize Backbone
        self.backbone, self.pruning_loc, global_feature_dim, num_patches, num_stages = create_global_branch(self.arch,
                                                                                                            cfg={
                                                                                                                **model_cfg})
        if self.arch.startswith("swin"):
            assert num_stages == 4
            C = global_feature_dim // 8
            global_feature_dim_list = [C * 2, C * 4, C * 8, C * 8]
            if model_cfg["image_size"] == 224:
                num_patches_list = [784, 196, 49, 49]
            elif model_cfg["image_size"] == 448:
                num_patches_list = [3136, 784, 196, 196]
            else:
                raise NotImplementedError
        else:
            global_feature_dim_list = [global_feature_dim] * num_stages
            num_patches_list = [num_patches] * num_stages

        self.feature_stage = num_stages
        self.local_feat_dim = global_feature_dim_list[self.feature_stage - 1]

        # Local Branch: Patch Graph Aggregator
        self.Batch_Branch = PatchGraphAggregator(self.local_feat_dim,
                                                 topk=local_cfg["topk"],
                                                 depth=local_cfg["depth"])
        self.local_proj_layer = nn.Linear(self.local_feat_dim, global_feature_dim)
        # Adaptive Feature Fusion Module
        self.weight_learning_module = DynamicFusionModule(2 * global_feature_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(global_feature_dim, model_cfg["num_classes"])
        )
        self.softmax_layer = nn.LogSoftmax(dim=1)

        self.cbam_reduction_ratio = local_cfg.get("cbam_reduction")
        self.cbam_kernel_size = local_cfg.get("cbam_kernel")
        self.top_patches = local_cfg["top_patches"]

        # Attention Mechanism for patch selection
        self.patch_cbam = CBAM(
            gate_channels=self.local_feat_dim,
            reduction_ratio=self.cbam_reduction_ratio,
            kernel_size=self.cbam_kernel_size
        )

        self.feature_aligner = AdaptiveWeighting(feature_dim=global_feature_dim, alpha=0.1, beta=0.1)

    def get_param_groups(self):
        # Separate backbone and head parameters for different learning rates
        param_groups = [[], []]
        for name, param in super().named_parameters():
            if name.startswith("backbone") and "score_predictor" not in name:
                param_groups[0].append(param)
            else:
                param_groups[1].append(param)
        return param_groups

    def fuse_feature(self, global_feature, local_feature=None, NoAFF=False):
        # Fuse global and local features using learned weights or simple addition
        alpha, beta = torch.ones(1), torch.ones(1)
        feature = global_feature
        if local_feature is not None:
            local_feature = self.local_proj_layer(local_feature)
            if not NoAFF:
                alpha, beta = self.weight_learning_module(global_feature, local_feature)
                feature = alpha * global_feature + beta * local_feature
            else:
                feature = global_feature + local_feature
        return feature


    def forward(self, images, targets=None, flag=None):

        # 1. Backbone Forward Pass
        if self.arch.startswith("swin"):
            global_features, global_patch_features, decision_mask, decision_mask_list, feature_list = self.backbone.forward_features(
                images)

        else:
            global_features, global_patch_features, global_attentions, decision_mask, decision_mask_list, feature_list = self.backbone.forward_features(
                images)

        local_features = None
        if not self.NoGNN:
            # 2. Local Branch Processing
            raw_local_patches = feature_list[self.feature_stage - 1]
            B = raw_local_patches.shape[0]

            if raw_local_patches.dim() == 3 and raw_local_patches.shape[1] == self.local_feat_dim:
                raw_local_patches = raw_local_patches.permute(0, 2, 1)
            B, N, C = raw_local_patches.shape

            H_local = W_local = int(math.sqrt(N))
            assert H_local * W_local == N, f"Patch number {N} must be a perfect square (check feature stage {self.feature_stage})"

            # Prepare mask from backbone decision
            patch_importance_mask = None
            if decision_mask is not None:
                if decision_mask.dim() == 3:
                    L_mask = decision_mask.shape[1]
                    H_mask = W_mask = int(math.sqrt(L_mask))
                    decision_mask_spatial = decision_mask.permute(0, 2, 1).view(B, decision_mask.shape[2], H_mask,
                                                                                W_mask)
                else:
                    raise ValueError(f"Unsupported decision_mask dim: {decision_mask.dim()}")

                decision_mask_dowmsampled = F.interpolate(
                    decision_mask_spatial,
                    size=(H_local, W_local),
                    mode='bilinear',
                    align_corners=False
                )
                patch_importance_mask = decision_mask_dowmsampled.mean(dim=1).view(B, N)

            # Apply CBAM attention
            weighted_patches, patch_scores = self.patch_cbam(raw_local_patches, patch_importance_mask)

            # Select Top-K Patches
            if self.NoMAPS:
                selected_patches = weighted_patches[:, :self.top_patches, :]
            else:
                topk_indices = torch.topk(patch_scores, k=self.top_patches, dim=1)[1]
                topk_indices_exp = topk_indices.unsqueeze(-1).expand(B, self.top_patches, C)
                selected_patches = torch.gather(weighted_patches, dim=1, index=topk_indices_exp)

            # Graph Aggregation
            local_features = self.Batch_Branch(selected_patches)

        # 3. Feature Fusion & Classification
        features = self.fuse_feature(global_feature=global_features, local_feature=local_features, NoAFF=self.NoAFF)
        logits = self.classifier(features)

        # 4. Return results (Training vs Inference)
        if self.training and not self.NoPFE:
            # Pairwise Feature Enhancer Loss calculation
            self_scores, other_scores = Pairwise_Feature_Enhancer(
                features=features,
                global_features=global_features,
                targets=targets,
                feature_aligner=self.feature_aligner,
                classifier=self.classifier,
                softmax_layer=self.softmax_layer
            )
            return logits, self_scores, other_scores, global_features, decision_mask_list
        else:
            if self.training and self.NoPFE:
                return logits, global_features, decision_mask_list
            else:
                return logits