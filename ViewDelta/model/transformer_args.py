from dataclasses import dataclass, asdict, field
from typing import Optional
from torch import optim
from torch.optim.optimizer import Optimizer
from typing import Type


@dataclass
class TransformerModelArgs:
    # these set architecture parameters
    # pass these to the datamodule since we don't train on them
    image_embeddings: str = (
        "dinov2"  # "dinov2" or "depth-anything-2"or "patch-embedding"
    )
    text_embeddings: str = "siglip"  # "siglip" or "clip"
    use_multiscale: bool = True  # use multiscale features for segmentation head or not
    use_separation_tokens: bool = (
        True  # use tokens to seperate image, text, and segmentation queries
    )
    multiscale_indices: Optional[list] = field(default_factory=lambda: [1, 2, 4, 6])
    img_tokens: int = 257  # tokens 257 for DINOv2, 1370 for depth anything 2
    text_tokens: int = 64  # tokens 64 for SigLip, 77 for CLIP

    # transformer parameters, should be changed based on architecture
    num_classes: int = 2  # binary segmentations

    # if using patch embedding
    patch_size: int = 16
    stride: int = 16

    image_embedding_dim: int = (
        768  # DINOv2 768 or depth anything 2 embedding dimension 1024, also dim for transformer
    )
    text_embedding_dim: int = 1024  # 1024 SigLip text embedding dimension 512 for CLIP
    dim: int = 512  # dimension of tokens used in attention layers of the transformer
    depth: int = 8
    heads: int = 8
    mlp_dim: Optional[int] = None  # Defaults to 4 * dim if not provided
    feature_map_size: int = (
        64  # height and width of the feature map that will be used by the segmentation head
    )
    dim_head: int = 64
    dropout: float = 0.00
    emb_dropout: float = 0.00
    checkpoint_ff: bool = False
    checkpoint_attn: bool = False

    # training parameters
    image_size: int = 256
    max_batch_size: int = 1
    output_dir: str = "training_artifacts"
    accumulate_grad_batches: int = 1
    alpha: float = 0.25
    gamma: float = 2
    log_freq: int = 100
    learning_rate: float = 1e-4
    optimizer: Type[Optimizer] | str = optim.Adam

    # optimizer: Any = DeepSpeedCPUAdam
    dice_average: str = "macro"
    epochs: int = 50
    warmup_epochs: int = 1
    output_label_size: int = 256

    use_gpu_feature_extraction: bool = False
    skip_empty_training_labels: bool = True

    # data parameters
    data_dir: str = "data"  # parent directory of the datasets
    dataset_names: list[str] = field(
        default_factory=lambda: ["PSCD", "sysu_cd", "cseg"]
    )

    validation_dataset_names: list[str] = field(
        default_factory=lambda: ["PSCD", "sysu_cd", "cseg"]
    )

    project_name: str = "dino_feats"  # logger project name
    job_id: str = "0"  # SLURM job ID
    skip_seq_tokens: bool = False
    use_seg_queries: bool = True  # whether to use segmentation queries or skip them

    nodes: int = 1
    devices: int = 1

    skip_image_feature_cache: bool = False
    feature_cache_dir: str = "image_feature_cache"  # directory to store cached features
    checkpoint_path: Optional[str] = None  # optional path to checkpoint file

    def __post_init__(self):
        if self.mlp_dim is None:
            self.mlp_dim = self.dim * 4
        assert self.dim % self.heads == 0

    def dict_convert(self):
        return asdict(self)
