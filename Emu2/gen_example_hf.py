import cv2
from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig
from emu.constants import *
from torchvision import transforms as TF


transform = TF.Compose([
            TF.Resize((EVA_IMAGE_SIZE, EVA_IMAGE_SIZE), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ])

# For the first time of using,
# you need to download the huggingface repo "BAAI/Emu2-GEN" to local first
path = "ckpts/Emu2-Gen"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

multimodal_encoder = AutoModelForCausalLM.from_pretrained(
    f"{path}/multimodal_encoder",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    # load_in_8bit=True,
    # bnb_4bit_compute_dtype=torch.float16,
    variant="bf16",
)

tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")

pipe = DiffusionPipeline.from_pretrained(
    path,
    custom_pipeline="pipeline_emu2_gen",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
    multimodal_encoder=None,
    tokenizer=tokenizer,
)

# For the non-first time of using, you can init the pipeline directly
# pipe = DiffusionPipeline.from_pretrained(
#     path,
#     custom_pipeline="pipeline_emu2_gen",
#     torch_dtype=torch.bfloat16,
#     use_safetensors=True,
#     variant="bf16",
# )

pipe.to("cuda")

# text-to-image
# prompt = "impressionist painting of an astronaut in a jungle"
# ret = pipe(prompt)
# ret.image.save("astronaut.png")

# # image editing
# image = Image.open("./examples/dog.jpg").convert("RGB")
# prompt = [image, "wearing a red hat on the beach."]
# ret = pipe(prompt)
# ret.image.save("dog_hat_beach.png")

# # grounding generation
# def draw_box(left, top, right, bottom):
#     mask = np.zeros((448, 448, 3), dtype=np.uint8)
#     mask = cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), 3)
#     mask = Image.fromarray(mask)
#     return mask

# dog1 = Image.open("./examples/dog1.jpg").convert("RGB")
# dog2 = Image.open("./examples/dog2.jpg").convert("RGB")
# dog3 = Image.open("./examples/dog3.jpg").convert("RGB")
# dog1_mask = draw_box( 22,  14, 224, 224)
# dog2_mask = draw_box(224,  10, 448, 224)
# dog3_mask = draw_box(120, 264, 320, 438)

# prompt = [
#     "<grounding>",
#     "An oil painting of three dogs,",
#     "<phrase>the first dog</phrase>"
#     "<object>",
#     dog1_mask,
#     "</object>",
#     dog1,
#     "<phrase>the second dog</phrase>"
#     "<object>",
#     dog2_mask,
#     "</object>",
#     dog2,
#     "<phrase>the third dog</phrase>"
#     "<object>",
#     dog3_mask,
#     "</object>",
#     dog3,
# ]
# ret = pipe(prompt)
# ret.image.save("three_dogs.png")

# # Autoencoding
# # to enable the autoencoding mode, please pull the latest pipeline_emu2_gen.py first,
# # and you can only input exactly one image as prompt
# # if you want the model to generate an image,
# # please input extra empty text "" besides the image, e.g.
# #   autoencoding mode: prompt = image or [image]
# #   generation mode: prompt = ["", image] or [image, ""]
# prompt = Image.open("./examples/doodle.jpg").convert("RGB")
# ret = pipe(prompt)
# ret.image.save("doodle_ae.png")

   
exo1 = Image.open("/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/preprocessed_episodes/train/fair_cooking_06_2/20/cam04.png").convert("RGB")
rgb1 = Image.open("/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/preprocessed_episodes/train/fair_cooking_06_2/20/ego_rgb.png").convert("RGB")
caption1 = 'Squeezes the pack of noodles with his hands.'
exo2 = Image.open("/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/preprocessed_episodes/train/fair_cooking_06_2/1247/cam04.png").convert("RGB")
rgb2 = Image.open("/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/preprocessed_episodes/train/fair_cooking_06_2/1247/ego_rgb.png").convert("RGB")
caption2 = 'Picks a knife from the chopping board with his right hand.'
exo3 = Image.open("/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/preprocessed_episodes/train/fair_cooking_06_2/1485/cam04.png").convert("RGB")
rgb3 = Image.open("/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/preprocessed_episodes/train/fair_cooking_06_2/1485/ego_rgb.png").convert("RGB")
caption3 = 'Moves the spring onion on the chopping board with his left hand.'
exo4 = Image.open("/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/preprocessed_episodes/train/fair_cooking_06_2/89/cam04.png").convert("RGB")
rgb4 = Image.open("/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/preprocessed_episodes/train/fair_cooking_06_2/89/ego_rgb.png").convert("RGB")
caption4 = 'Picks a knife from the chopping board with his right hand.'
prompt = [exo1, exo2, exo3, exo4, caption1]

text_prompt, image_prompt = "", []
for x in prompt:
    if isinstance(x, str):
        text_prompt += x
    else:
        text_prompt += DEFAULT_IMG_PLACEHOLDER
        image_prompt.append(transform(x))

if len(image_prompt) == 0:
    image_prompt = None
else:
    image_prompt = torch.stack(image_prompt)
    image_prompt = image_prompt.to(device=multimodal_encoder.model.device, dtype=multimodal_encoder.model.dtype)

has_image, has_text = True, True
negative_prompt = {}
# Enable Autoencoding Mode, you can ONLY input exactly one image

prompt = multimodal_encoder.generate_image(text=[text_prompt], image=image_prompt, tokenizer=tokenizer)

key = ""
if key not in negative_prompt:
    negative_prompt[key] = multimodal_encoder.generate_image(text=[key], tokenizer=tokenizer)
prompt = torch.cat([prompt, negative_prompt[key]], dim=0)

# prompt_embeds = multimodal_encoder.generate_image([text_prompt], tokenizer, image_prompt)
ret = pipe.validation_forward(prompt_embeds=prompt).image
ret.save("cooking_4bit_2stage_cfg_448.png")

# image = Image.open("/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/preprocessed_episodes/train/fair_cooking_06_2/20/cam04.png").convert("RGB")
# prompt = ["Ego view of cooking in the ",image]
# ret = pipe(prompt)
# ret.image.save("cooking.png")