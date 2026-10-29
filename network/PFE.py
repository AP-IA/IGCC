import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWeighting(nn.Module):
    def __init__(self, feature_dim, alpha=0.1, beta=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, anchor_feat, pair_feat, is_positive):
        """
        is_positive: Boolean mask, True for intra-class , False for inter-class
        """
        anchor_norm = self.norm(anchor_feat)
        pair_norm = self.norm(pair_feat)
        cos_sim = F.cosine_similarity(anchor_norm, pair_norm, dim=-1).unsqueeze(-1)
        pull_direction = pair_norm - anchor_norm
        pull_weight = (1 - cos_sim) * self.alpha
        push_direction = anchor_norm - pair_norm
        push_weight = torch.relu(cos_sim) * self.beta
        mask = is_positive.view(-1, 1).float()
        delta = mask * (pull_weight * pull_direction) + (1 - mask) * (push_weight * push_direction)
        enhanced_anchor = anchor_feat + delta
        return enhanced_anchor


def get_pairs(embeddings, labels):

    def pdist(vectors):
        distance_matrix = -2 * vectors.mm(torch.t(vectors)) + \
                          vectors.pow(2).sum(dim=1).view(1, -1) + \
                          vectors.pow(2).sum(dim=1).view(-1, 1)
        return distance_matrix

    distance_matrix = pdist(embeddings)

    labels = labels.to(embeddings.device).view(-1, 1)
    batch_size = embeddings.shape[0]

    same_class = labels == labels.t()
    positive_mask = same_class.clone()
    positive_mask.fill_diagonal_(False)
    negative_mask = ~same_class

    # A ranking comparison needs both a non-self positive and a true negative.
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    anchors = torch.arange(batch_size, device=embeddings.device)[valid]
    dist_same = distance_matrix[valid].masked_fill(~positive_mask[valid], float("inf"))
    dist_diff = distance_matrix[valid].masked_fill(~negative_mask[valid], float("inf"))
    intra_idxs = dist_same.argmin(dim=1)
    inter_idxs = dist_diff.argmin(dim=1)

    intra_labels = torch.cat([labels[anchors], labels[intra_idxs]], dim=1)
    inter_labels = torch.cat([labels[anchors], labels[inter_idxs]], dim=1)
    intra_pairs = torch.stack([anchors, intra_idxs], dim=1)
    inter_pairs = torch.stack([anchors, inter_idxs], dim=1)

    return intra_pairs, inter_pairs, intra_labels, inter_labels


def Pairwise_Feature_Enhancer(features, global_features, targets, feature_aligner, classifier, softmax_layer):

    with torch.no_grad():
        intra_pairs, inter_pairs, intra_labels, inter_labels = get_pairs(global_features, targets)

    if intra_pairs.shape[0] == 0:
        # Preserve a differentiable empty result; the caller skips ranking loss.
        empty_scores = features[:0].sum(dim=1)
        return empty_scores, empty_scores

    anchors = torch.cat([features[intra_pairs[:, 0]], features[inter_pairs[:, 0]]], dim=0)
    pairs = torch.cat([features[intra_pairs[:, 1]], features[inter_pairs[:, 1]]], dim=0)

    num_intra = intra_pairs.shape[0]
    num_inter = inter_pairs.shape[0]

    is_positive = torch.cat([
        torch.ones(num_intra, device=features.device, dtype=torch.bool),
        torch.zeros(num_inter, device=features.device, dtype=torch.bool)
    ], dim=0)

    enhanced_anchors = feature_aligner(anchors, pairs, is_positive)

    logits = classifier(enhanced_anchors)

    target_labels = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)

    scores = softmax_layer(logits)
    conf_scores = scores[torch.arange(scores.shape[0], device=scores.device), target_labels.to(torch.long)]

    score_intra = conf_scores[:num_intra]
    score_inter = conf_scores[num_intra:]

    return score_intra, score_inter
