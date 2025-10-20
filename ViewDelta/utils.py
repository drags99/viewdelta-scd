from PIL import Image


def load_image(image_path):
    return Image.open(image_path)


def resize_image(image, image_size):
    return image.resize((image_size, image_size))
