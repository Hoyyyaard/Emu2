from emu.diffusion import EmuVisualGeneration
import torch
from PIL import Image
import cv2
import numpy as np

pipe = EmuVisualGeneration.from_pretrained(
        "ckpts/Emu2-Gen/multimodal_encoder",
        dtype=torch.bfloat16,
        use_safetensors=True,
)

# Single GPU, e.g. cuda:0
pipe = pipe.multito(["cuda:0"])
# Multi GPU, e.g. cuda:0 and cuda:1
# pipe = pipe.multito(["cuda:0", "cuda:1"])

# text-to-image
# prompt = "impressionist painting of an astronaut in a jungle"
# ret = pipe(prompt)
# ret.image.save("astronaut.png")

# image editing
image = Image.open("./examples/dog.jpg").convert("RGB")
prompt = [image, "wearing a red hat on the beach."]
ret = pipe(prompt)
ret.image.save("dog_hat_beach.png")

# grounding generation
def draw_box(left, top, right, bottom):
    mask = np.zeros((448, 448, 3), dtype=np.uint8)
    mask = cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), 3)
    mask = Image.fromarray(mask)
    return mask

dog1 = Image.open("./examples/dog1.jpg").convert("RGB")
dog2 = Image.open("./examples/dog2.jpg").convert("RGB")
dog3 = Image.open("./examples/dog3.jpg").convert("RGB")
dog1_mask = draw_box( 22,  14, 224, 224)
dog2_mask = draw_box(224,  10, 448, 224)
dog3_mask = draw_box(120, 264, 320, 438)

prompt = [
    "<grounding>",
    "An oil painting of three dogs,",
    "<phrase>the first dog</phrase>"
    "<object>",
    dog1_mask,
    "</object>",
    dog1,
    "<phrase>the second dog</phrase>"
    "<object>",
    dog2_mask,
    "</object>",
    dog2,
    "<phrase>the third dog</phrase>"
    "<object>",
    dog3_mask,
    "</object>",
    dog3,
]
ret = pipe(prompt)
ret.image.save("three_dogs.png")

# Autoencoding
# to enable the autoencoding mode, you can only input exactly one image as prompt
# if you want the model to generate an image,
# please input extra empty text "" besides the image, e.g.
#   autoencoding mode: prompt = image or [image]
#   generation mode: prompt = ["", image] or [image, ""]
prompt = Image.open("./examples/doodle.jpg").convert("RGB")
ret = pipe(prompt)
ret.image.save("doodle_ae.png")