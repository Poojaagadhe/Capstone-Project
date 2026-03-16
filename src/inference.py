import torch
from src.model import UNetGenerator


def load_model():

    model = UNetGenerator()

    checkpoint = torch.load(
        "models/best_model_G.pth",
        map_location=torch.device("cpu")
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def generate_ct(model, image_tensor):

    with torch.no_grad():
        output = model(image_tensor)

    return output
