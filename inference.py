import torch
from ViewDelta import embedders
from ViewDelta.model.transformer_args import TransformerModelArgs
from ViewDelta.embedders import get_embedders, get_model_features_from_image
from ViewDelta.utils import load_image, resize_image
import matplotlib.pyplot as plt
import numpy as np
from ViewDelta.model.model_feature_segmentor import TextConditionedDecoder
import os


def get_segmentation_mask(image_a, image_b, text, model, embedders, model_args):
    with torch.no_grad():
        # load and resize images to 256x256
        image_a = load_image(image_a)
        image_b = load_image(image_b)
        image_a = resize_image(image_a, 256)
        image_b = resize_image(image_b, 256)

        # get features
        image_a_features = get_model_features_from_image(
            image_a, embedders["image_model"], embedders["image_processor"], model_args
        )
        image_b_features = get_model_features_from_image(
            image_b, embedders["image_model"], embedders["image_processor"], model_args
        )
        text_tokens = embedders["text_processor"](
            text=text, padding="max_length", return_tensors="pt"
        )
        text_features = embedders["text_model"](**text_tokens)[
            "last_hidden_state"
        ].detach()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_a_features = image_a_features.to(device)
        image_b_features = image_b_features.to(device)
        text_features = text_features.to(device)
        model = model.to(device)
        # run model
        output = model(image_a_features, image_b_features, text_features)

        # get segmentation mask using argmax
        segmentation_mask = torch.argmax(output, dim=1).detach().cpu().numpy()
        return segmentation_mask


if __name__ == "__main__":

    image_a_list = ["assets/house_image_a.png", "assets/house_image_a.png", "assets/construction_image_a.png"]
    image_b_list = ["assets/house_image_b.png", "assets/house_image_b.png", "assets/construction_image_b.png"]
    text_list = ["concrete", "garage", "bulldozer"]

    # path to the checkpoint to be used
    PATH_TO_CHECKPOINT = "viewdelta_checkpoint.pth"

    model_args = TransformerModelArgs(
        text_embeddings="siglip",
        image_embeddings="dinov2",
        use_multiscale=False,
        use_separation_tokens=False,
        depth=12,
        dim=768,
        mlp_dim=3072,
        heads=12,
        checkpoint_attn=False,
        checkpoint_ff=False,
    )

    if model_args.depth == 8:
        multiscale_indices = [1, 2, 4, 7]
    elif model_args.depth == 12:
        multiscale_indices = [1, 3, 6, 11]
    elif model_args.depth > 12:
        multiscale_indices = [round(i * (model_args.depth - 1) / 3) for i in range(4)]

    if model_args.text_embeddings == "siglip":
        model_args.text_tokens = 64
        model_args.text_embedding_dim = 1024
    elif model_args.text_embeddings == "clip":
        model_args.text_tokens = 77
        model_args.text_embedding_dim = 768
    else:
        assert False, "Text option not supported"

    if model_args.image_embeddings == "dinov2":
        model_args.img_tokens = 257
        model_args.image_embedding_dim = 1024
    elif model_args.image_embeddings == "depth-anything-2":
        model_args.img_tokens = 1370
        model_args.image_embedding_dim = 1024
    else:
        assert False, "Image option not supported"

    model = TextConditionedDecoder(model_args)
    state_dict = torch.load(PATH_TO_CHECKPOINT, map_location=torch.device("cuda:0"))
    model.load_state_dict(state_dict)

    embedders = get_embedders(model_args)
    text_embedder = embedders["text_model"]
    text_processor = embedders["text_processor"]

    image_embedder = embedders["image_model"]
    image_processor = embedders["image_processor"]

    for image_a, image_b, text in zip(image_a_list, image_b_list, text_list):
        image_name = image_a.split(".")[0]

        if "/" in image_a:
            image_name = image_a.split("/")[-1]
        else:
            image_name = image_a.split(".")[0]
        segmentation_mask = get_segmentation_mask(
            image_a, image_b, text, model, embedders, model_args
        )

        # resize image_a to match the segmentation mask and overlay the segmentation mask on it
        image_a = load_image(image_a)
        image_a = resize_image(image_a, 256)

        # Convert image to numpy
        image_a = np.array(image_a)

        # Normalize mask to 0–1
        mask = segmentation_mask[0].astype(np.float32)

        # Handle RGB or RGBA
        num_channels = image_a.shape[2]
        overlay_color = np.array([255, 0, 0, 0], dtype=np.uint8)

        # Create overlay with the same number of channels
        overlay = np.zeros_like(image_a, dtype=np.uint8)
        overlay[:] = overlay_color

        # Blend overlay and original image (alpha controls transparency)
        alpha = 0.5
        overlayed_image = np.where(
            mask[..., None] > 0.5,
            (alpha * overlay + (1 - alpha) * image_a).astype(np.uint8),
            image_a
        )

        plt.imshow(overlayed_image)
        plt.axis("off")
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/{image_name}_{text}_image_a_overlay.png", bbox_inches="tight", pad_inches=0)
        plt.close()

