from transformers import (
    AutoProcessor,
    AutoModel,
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)
from ViewDelta.model.transformer_args import TransformerModelArgs
import torch
import torch.cuda.amp as amp


# use to get siglip or clip openai/clip-vit-base-patch32,openai/clip-vit-large-patch14,google/siglip-large-patch16-384"
def get_text_model(model_name="google/siglip-large-patch16-384"):
    model = AutoModel.from_pretrained(model_name, cache_dir=".")
    processor = AutoProcessor.from_pretrained(model_name)
    text_model = model.text_model
    return text_model, processor


# use to get dinov2
def get_image_model(model_name="facebook/dinov2-large"):
    model = AutoModel.from_pretrained(model_name, cache_dir=".")
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


# use to get depth anythingv2
def get_depth_model(model_name="depth-anything/Depth-Anything-V2-Large-hf"):
    model = AutoModelForDepthEstimation.from_pretrained(model_name, cache_dir=".")
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


# get dinov2 features from image
def get_model_features_from_image(image, model, processor, args):
    # send input to device and model to device if args.use_gpu_feature_extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.image_embeddings.lower() == "dinov2":
        inputs = processor(images=[image] if not isinstance(image, list) else image, return_tensors="pt")
        with torch.no_grad():

            if args.use_gpu_feature_extraction:
                model = model.to(device)
                inputs = {
                    k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }
                with amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        features = outputs.last_hidden_state.detach().to(
            "cpu"
        )  # shape: (B, img_tokens, dinov2_dim)
    elif args.image_embeddings.lower() in ["depth-anything-2", "depth-anything"]:
        inputs = processor(images=[image] if not isinstance(image, list) else image, return_tensors="pt")

        if args.use_gpu_feature_extraction:
            model = model.to(device)
            inputs = {
                k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }
            with amp.autocast():
                outputs = model(**inputs, output_hidden_states=True)
        else:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
        features = (
            outputs["hidden_states"][-1].detach().to("cpu")
        )  # shape: (B, img_tokens, depth_dim)
    return features


def get_embedders(args: TransformerModelArgs):
    # Select text model
    if args.text_embeddings.lower() == "siglip":
        text_model, text_processor = get_text_model("google/siglip-large-patch16-384")
    elif args.text_embeddings.lower() == "clip":
        text_model, text_processor = get_text_model("openai/clip-vit-large-patch14")
    else:
        raise ValueError(f"Unsupported text_embeddings: {args.text_embeddings}")

    # Select image model
    if args.image_embeddings.lower() == "dinov2":
        image_model, image_processor = get_image_model("facebook/dinov2-large")
    elif args.image_embeddings.lower() in ["depth-anything-2", "depth-anything"]:
        image_model, image_processor = get_depth_model(
            "depth-anything/Depth-Anything-V2-Large-hf"
        )
    elif args.image_embeddings.lower() == "patch-embedding":
        image_model, image_processor = torch.nn.Identity(), torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported image_embeddings: {args.image_embeddings}")

    return {
        "text_model": text_model,
        "text_processor": text_processor,
        "image_model": image_model,
        "image_processor": image_processor,
    }
