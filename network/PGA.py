import torch
from torch import nn
import torch.nn.functional as F



class dot_productKNNGraphConv(nn.Module):
    def __init__(self, feature_dim, topk):
        super(dot_productKNNGraphConv, self).__init__()

        self.k_neighbors = topk
        self.feedforward = nn.Sequential(
            nn.Linear(feature_dim * topk, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )

    def forward(self, input_feats):  # input_feats: B, N, C

        batch_size, num_patches, channels = input_feats.shape
        transposed_feats = input_feats.permute(1, 0, 2)  # N, B, C

        dot_product_matrix = transposed_feats.bmm(transposed_feats.permute(0, 2, 1).contiguous())  # N, B, B

        actual_k = min(self.k_neighbors + 1, dot_product_matrix.shape[-1])
        top_weights, top_indices = torch.topk(dot_product_matrix, k=actual_k, dim=-1)
        actual_neighbors = actual_k - 1
        top_weights = top_weights[:, :, 1:1 + actual_neighbors]  # N, B, actual_K
        top_indices = top_indices[:, :, 1:1 + actual_neighbors]  # N, B, actual_K

        top_indices = top_indices.to(torch.long)

        patch_range = torch.arange(num_patches).view(-1, 1, 1).to(top_indices.device)  # shape: [N, 1, 1]
        neighbor_feats = transposed_feats[patch_range, top_indices, :]  # N, B, actual_K, C

        normalized_weights = F.softmax(top_weights, dim=2)
        weighted_neighbors = torch.mul(normalized_weights.unsqueeze(-1), neighbor_feats)  # N, B, actual_K, C

        if actual_neighbors < self.k_neighbors:
            pad_size = self.k_neighbors - actual_neighbors
            pad = torch.zeros(num_patches, batch_size, pad_size, channels, dtype=weighted_neighbors.dtype,
                              device=weighted_neighbors.device)
            weighted_neighbors = torch.cat([weighted_neighbors, pad], dim=2)

        aggregated = self.feedforward(weighted_neighbors.view(num_patches, batch_size, -1))  # N, B, C

        aggregated = aggregated.permute(1, 0, 2)

        return aggregated


class PatchGraphAggregator(nn.Module):
    def __init__(self, in_channels, depth, topk):
        super(PatchGraphAggregator, self).__init__()

        self.batch_size_padding = 3

        hidden_dim = 256

        self.downsample = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim)
        )

        blocks = []
        for i in range(depth):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                dot_productKNNGraphConv(feature_dim=hidden_dim, topk=topk),
                nn.Linear(hidden_dim, hidden_dim)
            )
            blocks.append(block)
        self.graph_blocks = nn.ModuleList(blocks)

        self.upsample = nn.Sequential(
            nn.Linear(hidden_dim, in_channels, bias=False),
            nn.LayerNorm(in_channels)
        )


    def forward(self, x):

        B, N, C = x.shape
        patch_feats = x
        shortcut = patch_feats.mean(dim=1) # BxC
        patch_feats = self.downsample(patch_feats)  # B, N, C

        ## batch
        padding = False
        if patch_feats.shape[0] < self.batch_size_padding:
            padding = True
        if padding:
            original_batch_size = B
            patch_feats = torch.cat([patch_feats, torch.zeros((self.batch_size_padding-patch_feats.shape[0],*patch_feats.shape[1:]), dtype=patch_feats.dtype, device=patch_feats.device)], dim=0)

        for block in self.graph_blocks:
            patch_feats = block(patch_feats)
        if padding:
            patch_feats = patch_feats[:original_batch_size]

        patch_feats = self.upsample(patch_feats)  # B, N, C

        patch_feats = patch_feats.mean(dim=1)
        patch_feats += shortcut

        return patch_feats