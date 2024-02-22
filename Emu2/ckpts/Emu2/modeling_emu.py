from functools import partial
from typing import Any, List, Optional, Mapping, Callable
from collections import OrderedDict
from argparse import Namespace
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import PIL
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from .configuration_emu import EmuConfig
from .constants import *
from .modeling_llama import LlamaForCausalLM
from .visual import EVAVisionTransformer


class EmuPreTrainedModel(PreTrainedModel):
    config_class = EmuConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayer", "Block"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class EmuForClsAndRegression(EmuPreTrainedModel):

    def __init__(self, config):
        super(EmuForClsAndRegression, self).__init__(config)

        self.lm = LlamaForCausalLM(config=config)

        self.lm.model.embed_tokens.padding_idx = config.pad_token_id

    def get_num_layers(self):
        return len(self.lm.model.layers)

class EmuModel(EmuPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        vision_config = Namespace(**config.vision_config)

        self.visual = EVAVisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.width // vision_config.head_width,
            mlp_ratio=vision_config.mlp_ratio,
            qkv_bias=vision_config.qkv_bias,
            drop_path_rate=vision_config.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=vision_config.layer_norm_eps),
            xattn=vision_config.xattn,
            postnorm=vision_config.postnorm,
        )

        self.decoder = EmuForClsAndRegression(config)

        self.gradient_checkpointing = False
        
        self.n_query = vision_config.n_query
        self.v_query = vision_config.v_query

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor, *, n_query=None):
        n_query = n_query if n_query is not None else self.n_query

        image_embeds = self.visual(image)
        image_embeds = image_embeds[:, 1:, :]
        b, n, c = image_embeds.shape
        sqrt_n = int(n**0.5)
        image_embeds = image_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)

        stride = int(sqrt_n // (n_query ** 0.5))
        image_embeds = F.avg_pool2d(image_embeds, kernel_size=(stride, stride), stride=stride)
        image_embeds = image_embeds.view(b, c, -1).permute(0, 2, 1).contiguous()
        return image_embeds


class EmuForCausalLM(EmuPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.model = EmuModel(config)
        # LM to EVA
        self.project_down = nn.Linear(config.hidden_size, config.d_model, bias=False)
        # EVA to LM
        self.project_up = nn.Linear(config.d_model, config.hidden_size, bias=False)

        self.n_query = self.model.n_query
        self.v_query = self.model.v_query

        self.image_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_IMAGE_TOKEN * self.n_query + DEFAULT_IMG_END_TOKEN
        # temporarily borrow [gIMG] as the video frame feature placeholder.
        self.video_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_gIMG_TOKEN * self.v_query + DEFAULT_IMG_END_TOKEN

    # @property
    # def device(self):
    #     return next(iter(self.parameters())).device

    # @property
    # def dtype(self):
    #     return next(iter(self.parameters())).dtype
    def device(self, module=None):
        if module is None:
            return next(self.parameters()).device
        return next(module.parameters()).device

    def dtype(self, module):
        if module is None:
            return next(self.parameters()).dtype
        return next(module.parameters()).dtype


    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=5,
        max_new_tokens=10,
        min_len=1,
        do_sample=False,
        penalty_alpha=None,
        top_p=None,
        top_k=None,
        temperature=None,
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs
    ):

        text_embeds = self.model.decoder.lm.model.embed_tokens(input_ids).to("cuda")
        if image is not None:
            prompt_image_embeds = self.model.encode_image(image, n_query=self.n_query)
            _, _, c = prompt_image_embeds.shape
            prompt_image_embeds = prompt_image_embeds.view(-1, c)
            prompt_image_embeds = self.project_up(prompt_image_embeds)
            image_idx = (input_ids == IMAGE)
            text_embeds[image_idx] = prompt_image_embeds.to(text_embeds.device)

        if video is not None:
            prompt_video_embeds = self.model.encode_image(video, n_query=self.v_query)
            _, _, c = prompt_video_embeds.shape
            prompt_video_embeds = prompt_video_embeds.view(-1, c)
            prompt_video_embeds = self.project_up(prompt_video_embeds)
            video_idx = (input_ids == VIDEO)
            text_embeds[video_idx] = prompt_video_embeds.to(text_embeds.device)

        outputs = self.model.decoder.lm.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_len,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            **kwargs,
        )

        return outputs

    def prepare_image_input(self, images):
        image_size: int = self.config.vision_config['image_size']
        transform = T.Compose(
            [
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
            ]
        )
        images = [transform(image) for image in images]
        return torch.stack(images, 0)

    def _prepare_chat_template(self, text, system_msg=""):
        text = [
            system_msg + USER_TOKEN + ": " + t + ASSISTANT_TOKEN +":"
            for t in text
        ]
        return text

    def prepare_text_input(
        self, 
        text: List[str],
        tokenizer: PreTrainedTokenizer,
        image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
        video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
        ):
        text = [
            t.replace(image_placeholder, self.image_placeholder).replace(video_placeholder, self.video_placeholder)
            for t in text
        ]
        input_ids = tokenizer(text, padding="longest", return_tensors="pt")
        return input_ids
        

    def build_input_ids(
            self,
            text: List[str],
            tokenizer: PreTrainedTokenizer,
            image: Optional[List["PIL.Image"]] = None,
            video: Optional[List["PIL.Image"]] = None,
            system_msg: str = "",
            to_cuda: bool = True
        ):

        if self.config.model_version == "chat":
            text = self._prepare_chat_template(text, system_msg)

        if image is not None:
            image = self.prepare_image_input(image)
        if video is not None:
            video = self.prepare_image_input(video)
        inputs = self.prepare_text_input(text, tokenizer)
        input_ids = inputs.input_ids
        attention_mask =  inputs.attention_mask

        if to_cuda:
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            if image is not None:
                image = image.to("cuda")
            if video is not None:
                video = video.to("cuda")


        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'video': video
        }

    def __call__(
        self,
        text: List[str],
        tokenizer: PreTrainedTokenizer,
        image: Optional[torch.Tensor] = None,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        IMAGE, BOI = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_IMG_TOKEN])
        if image is not None:
            batch_size = len(image)
            # Now image is a list of images in [(5,3,448,448), (...)]
            flattened_images = torch.cat(image, dim=0)
            # [bs*5, 3, 448, 448]
            prompt_image_embeds = self.model.encode_image(flattened_images)
            _, _, c = prompt_image_embeds.shape
            label = prompt_image_embeds
            label = label.view(batch_size, -1, self.n_query, c)
            label = label[:, -1, :, :].reshape(-1, c)
            
            prompt_image_embeds = prompt_image_embeds.view(-1, c)
            prompt_image_embeds = self.project_up(prompt_image_embeds)

        text = [t.replace(placeholder, self.image_placeholder) for t in text]

        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        device = self.device(self.model.decoder.lm.model.embed_tokens)
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device) # B x N
        text_embeds = self.model.decoder.lm.model.embed_tokens(input_ids)
        image_idx = (input_ids == IMAGE)
        text_embeds[image_idx] = prompt_image_embeds.to(text_embeds.device)

        outputs = self.model.decoder.lm.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        hidden_states = self.project_down(hidden_states)

        # Compute last image embedding positions
        boi_idx = torch.where(input_ids == BOI)
        last_img_boi_idx = []
        for bi in range(batch_size):
            last_img_boi_idx.append(boi_idx[1][boi_idx[0] == bi][-1].item())
        # Position of BOI of the input sequence will be the first img embedding position of the output sequence as forward will generate new token at the end of the sequence
        preds = []
        # targets = []
        for i, idx in enumerate(last_img_boi_idx):
            preds.append(hidden_states[i, idx:idx+self.n_query, :].contiguous())
            # targets.append(label[i, idx-1:idx+self.n_query, :].contiguous())

        preds = torch.stack(preds).view(-1, hidden_states.shape[-1])
        # targets = torch.stack(targets).view(-1, hidden_states.shape[-1])

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(preds, label)

        return loss


    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        tokenizer: PreTrainedTokenizer,
        image: Optional[torch.Tensor] = None,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        IMAGE, BOI = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_IMG_TOKEN])
        if image is not None:
            prompt_image_embeds = self.model.encode_image(image)
            _, _, c = prompt_image_embeds.shape
            prompt_image_embeds = prompt_image_embeds.view(-1, c)
            prompt_image_embeds = self.project_up(prompt_image_embeds)

        text = [t.replace(placeholder, self.image_placeholder) for t in text]

        target_image_embeds = None
        for num_img_token in range(self.n_query):
            if num_img_token == 0:
                text = [f"{t}{DEFAULT_IMG_TOKEN}" for t in text]
            else:
                text = [f"{t}{DEFAULT_IMAGE_TOKEN}" for t in text]

            inputs = tokenizer(text, padding="longest", return_tensors="pt")
            device = self.device(self.model.decoder.lm.model.embed_tokens)
            attention_mask = inputs.attention_mask.to(device)
            input_ids = inputs.input_ids.to(device) # B x N

            text_embeds = self.model.decoder.lm.model.embed_tokens(input_ids)

            image_idx = (input_ids == IMAGE)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            if image is not None:
                prompt_idx = torch.logical_and(image_idx, cumsum_idx > num_img_token)
                text_embeds[prompt_idx] = prompt_image_embeds.to(text_embeds.device)

            if target_image_embeds is not None:
                target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token))
                text_embeds[target_idx] = self.project_up(target_image_embeds).to(text_embeds.device)

            outputs = self.model.decoder.lm.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            image_idx = (input_ids == IMAGE) + (input_ids == BOI)
            cumsum_idx = torch.flip(torch.cumsum(torch.flip(image_idx, dims=[1]), dim=1), dims=[1])
            target_idx = torch.logical_and(image_idx, torch.logical_and(cumsum_idx > 0, cumsum_idx <= num_img_token+1))

            hidden_states = outputs.hidden_states[-1]
            target_image_embeds = hidden_states[target_idx.to(hidden_states.device)]
            target_image_embeds = target_image_embeds.view(-1, target_image_embeds.shape[-1])
            target_image_embeds = self.project_down(target_image_embeds)

        _, C = target_image_embeds.shape
        B = hidden_states.shape[0]
        target_image_embeds = target_image_embeds.view(B, -1, C)

        return target_image_embeds

