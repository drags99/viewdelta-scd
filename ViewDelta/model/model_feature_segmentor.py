import torch
from torch import nn
from einops import rearrange
from ViewDelta.model.transformer_args import TransformerModelArgs
import deepspeed
from ViewDelta.model.patch_embedder import PatchEmbedder
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RMSNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        # x: (b, c, h, w)
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)  # (b, 1, h, w)
        x = x / rms  # Normalize across channels
        return x * self.weight[:, None, None]  # Scale each channel


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        return_intermediate=False,
        checkpoint_ff=False,
        checkpoint_attn=False,
    ):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.checkpoint_ff = checkpoint_ff
        self.checkpoint_attn = checkpoint_attn

    def forward(self, x):
        intermediates = [] if self.return_intermediate else None
        for attn, ff in self.layers:
            if self.checkpoint_attn:
                x = deepspeed.checkpointing.checkpoint(attn, x) + x
            else:
                x = attn(x) + x
            if self.checkpoint_ff:
                x = deepspeed.checkpointing.checkpoint(ff, x) + x
            else:
                x = ff(x) + x
            if self.return_intermediate:
                intermediates.append(x)
        x = self.norm(x)
        if self.return_intermediate:
            intermediates.append(x)
            return x, intermediates
        return x


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = RMSNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = RMSNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = RMSNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        x_up = self.upsample(x)
        shortcut = self.shortcut(x_up)
        out = self.conv1(x_up)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.gelu(out + shortcut)


class MultiScaleFusion(nn.Module):
    def __init__(self, dim, num_scales):
        super().__init__()
        self.conv1 = nn.Conv2d(dim * num_scales, dim, kernel_size=1)
        self.bn1 = RMSNorm2d(dim)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = RMSNorm2d(dim)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.gelu(x)


class MultiScaleUpsamplingNetwork(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        # Upsample from 64x64 -> 256x256 using two ResidualUpBlocks.
        self.up1 = ResidualUpBlock(dim, dim)
        self.up2 = ResidualUpBlock(dim, dim)
        self.classifier = nn.Conv2d(dim, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        return self.classifier(x)


class TextConditionedDecoder(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.feature_map_size = args.feature_map_size
        self.num_seg_tokens = args.feature_map_size * args.feature_map_size
        self.img_tokens = args.img_tokens
        self.text_tokens = args.text_tokens
        self.skip_seq_tokens = args.skip_seq_tokens
        self.use_seg_queries = args.use_seg_queries
        self.args = args
        if self.args.image_embeddings.lower() == "patch-embedding":
            self.patch_embedder = PatchEmbedder(
                img_size=args.image_size,
                patch_size=args.patch_size,
                stride=args.stride,
                embed_dim=args.image_embedding_dim,
            )
        else:
            self.patch_embedder = nn.Identity()

        self.total_tokens = 2 * args.img_tokens + args.text_tokens
        if args.use_seg_queries:
            self.total_tokens += self.num_seg_tokens
        if args.use_separation_tokens:
            self.total_tokens += 3
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_tokens, args.dim))

        # Only create seg_queries if we're using them
        if args.use_seg_queries:
            self.seg_queries = nn.Parameter(
                torch.randn(1, self.num_seg_tokens, args.dim)
            )
        else:
            self.seg_queries = None

        # makes the text embedding the same dimension as the image embedding
        self.text_projection = nn.Sequential(
            nn.RMSNorm(args.text_embedding_dim),
            nn.Linear(args.text_embedding_dim, args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim),
        )

        self.image_projection = nn.Sequential(
            nn.RMSNorm(args.image_embedding_dim),
            nn.Linear(args.image_embedding_dim, args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim),
        )

        norm_dim = args.dim
        if args.use_seg_queries:
            norm_dim = args.dim
        
        self.norm = RMSNorm2d(norm_dim, eps=1e-5)
        self.pre_norm = RMSNorm(args.dim, eps=1e-5)

        self.img_a_norm = RMSNorm(args.dim, eps=1e-5)
        self.img_b_norm = RMSNorm(args.dim, eps=1e-5)
        self.text_norm = RMSNorm(args.dim, eps=1e-5)

        self.combined_projection = FeedForward(args.dim, args.dim * 3, dropout=0)

        self.img_a_seperation_token = nn.Parameter(torch.randn(1, 1, args.dim))
        self.img_b_seperation_token = nn.Parameter(torch.randn(1, 1, args.dim))
        self.text_seperation_token = nn.Parameter(torch.randn(1, 1, args.dim))

        self.dropout = nn.Dropout(args.emb_dropout)
        self.transformer = Transformer(
            args.dim,
            args.depth,
            args.heads,
            args.dim_head,
            args.mlp_dim,
            args.dropout,
            return_intermediate=args.use_multiscale,
            checkpoint_ff=args.checkpoint_ff,
            checkpoint_attn=args.checkpoint_attn,
        )
        self.use_multiscale = args.use_multiscale

        # Conditionally set input channels based on whether we use seg queries
        # When not using seg queries, we concatenate two image token sequences
        input_dim = args.dim #if self.use_seg_queries else 2 * args.dim

        self.upsampling_network = nn.Sequential(
                # Step 1: Refine features with a convolution (preserves spatial dims)
                # Input channels adjusted based on seg query usage
                nn.Conv2d(input_dim, 256, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                # Step 2: Upsample by 2 using a transposed convolution (or interpolation)
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=False),
                nn.ReLU(inplace=True),
                # Step 3: Further refinement
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                # Instead of a final transposed convolution, use bilinear upsampling...
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                # ...followed by a final convolution to output num_classes channels
                nn.Conv2d(256, args.num_classes, kernel_size=3, padding=1, bias=False),
            )
            # self.upsampling_network = nn.ConvTranspose2d(
            #     args.dim, args.num_classes, kernel_size=4, stride=4
            # )

    def forward(self, imga, imgb, text_embeddings):
        b = imga.shape[0]
        # remove the spare empty dimension
        if self.args.image_embeddings.lower() == "patch-embedding":
            imga = self.patch_embedder(imga)
            imgb = self.patch_embedder(imgb)
        else:
            imga = imga.squeeze(1)
            imgb = imgb.squeeze(1)
        text_embeddings = text_embeddings.squeeze(1)
        imga = self.image_projection(imga)
        imgb = self.image_projection(imgb)
        text_tokens = self.text_projection(text_embeddings)

        imga = self.img_a_norm(imga)
        imgb = self.img_b_norm(imgb)
        text_tokens = self.text_norm(text_tokens)

        b, _, c = imga.shape

        tokens_to_cat = [
            imga,
            imgb,
            text_tokens,
        ]
        if self.use_seg_queries and self.seg_queries is not None:
            tokens_to_cat.append(self.seg_queries.expand(b, -1, -1))
        x = torch.cat(tokens_to_cat, dim=1)

        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.combined_projection(x)
        x = self.pre_norm(x)

        if self.skip_seq_tokens:
            imga_tokens = x[:, : self.img_tokens-1, :]
            imgb_tokens = x[:, self.img_tokens : 2 * self.img_tokens-1, :]
            assert imga_tokens.shape == imgb_tokens.shape
            combined_img_tokens = imga_tokens + imgb_tokens
            feature_map_size = int((self.img_tokens) ** 0.5)

            seg_tokens = combined_img_tokens.view(
                b, feature_map_size, feature_map_size, c
            )
            seg_tokens = seg_tokens.permute(0, 3, 1, 2)
            seg_tokens = self.norm(seg_tokens)
            # upsample before upsampling network
            seg_tokens = F.interpolate(
                seg_tokens,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )
            seg_logits = self.upsampling_network(seg_tokens)
        else:
            x = self.transformer(x)
            if self.use_seg_queries:
                # Use segmentation query tokens from the transformer output
                seg_tokens = x[:, -self.num_seg_tokens :, :]
                map_size = int(self.num_seg_tokens**0.5)
                seg_tokens = seg_tokens.view(b, map_size, map_size, -1)
                seg_tokens = seg_tokens.permute(0, 3, 1, 2)
                seg_tokens = self.norm(seg_tokens)
                seg_tokens = F.interpolate(
                    seg_tokens,
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                )
                seg_logits = self.upsampling_network(seg_tokens)

        return seg_logits
