# Import pkg here
from PIL import Image 
import requests
import torch 
from emu.constants import *
import logging
import shutil
from torchvision import transforms as TF
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from diffusers import DiffusionPipeline
from accelerate import Accelerator
from accelerate import FullyShardedDataParallelPlugin
import PIL
from transformers import BitsAndBytesConfig
from torchvision import transforms
import math
from tqdm import tqdm
import datasets
import diffusers
import transformers
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
import argparse 
import os
from dataset import Diffusion_Finetune_Dataset
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict


# Argparse here
def parse_args():
    parser = argparse.ArgumentParser(description="Emu2-Gen finetuning")
    parser.add_argument("--model_name", type=str, default="Emu2-Gen", choices=['Emu2', 'Emu2-Gen'])
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution of the images")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-6, help="Adam epsilon")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report to")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for the dataloader")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--output_dir", type=str, default="results/Emu2-Gen/multimodal_encoder", help="Output directory")
    parser.add_argument("--resume_from_checkpoint", type=str, default='latest', help="Resume from checkpoint")
    parser.add_argument("--use_8bit_adam", action="store_true", default=True, help="Use 8-bit Adam")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'),)
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Learning rate warmup steps")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max train steps")
    parser.add_argument("--checkpoints_total_limit", type=int, default=2, help="Checkpoints total limit")
    # parser.add_argument("--center_crop", action="store_true", help="Center crop images")
    # parser.add_argument("--random_flip", action="store_true", help="Random flip images")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--checkpointing_steps", type=int, default=5, help="Checkpointing steps")
    parser.add_argument("--validation_epochs", type=int, default=5, help="Validation epochs")
    parser.add_argument("--val", action="store_true", help="Run validation")
    parser.add_argument("--use_quant_model", action="store_true", help="4 bit quant to load the model")
    parser.add_argument("--lora", action="store_true", help="")

    return parser.parse_args()

# Build dataset here
def build_dataloader(args, tokenizer):

    def tokenize_captions(captions):
        inputs = tokenizer(
            captions,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def collate_fn(examples):
        exo_pixel_values = torch.stack([example["exo_pixel_values"] for example in examples])
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.cat([example["input_ids"] for example in examples])
        text = [example['text'] for example in examples]
        image = [example['image'] for example in examples]
        origin_image = [example['original_image'] for example in examples]
        interleave_sequence = [example['interleave_sequence'] for example in examples]
        interleave_sequence_val = [example['interleave_sequence_val'] for example in examples]
        return {"edited_pixel_values": pixel_values, "input_ids": input_ids, 'image':image, 'original_image':origin_image, 'text':text, 'exo_pixel_values':exo_pixel_values, 'original_pixel_values':original_pixel_values, 'interleave_sequence':interleave_sequence, 'interleave_sequence_val':interleave_sequence_val}
    
    def preprocess_func(image, text):
        return train_transforms(image), tokenize_captions(text)
    print("Loading dataset...")
    train_dataset = Diffusion_Finetune_Dataset(preprocess_func=preprocess_func, split='train')
    val_dataset = Diffusion_Finetune_Dataset(preprocess_func=preprocess_func, split='val')

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    return train_dataloader, val_dataloader

# Build model here
def build_model(args):
    MODEL_CKPT_MAP = {'Emu2': 'ckpts/Emu2', 'Emu2-Gen': 'ckpts/Emu2-Gen/multimodal_encoder'}
    MODEL_PATH = MODEL_CKPT_MAP[args.model_name]

    tokenizer = AutoTokenizer.from_pretrained('ckpts/Emu2-Gen/tokenizer') 

    # with init_empty_weights():
    if args.use_quant_model:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map={'':Accelerator().process_index}
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={'':Accelerator().process_index}
            )  
    
    if args.lora:
        # lora config
        print("--------Apply Lora----------------")
        lora_config = LoraConfig(
            r = 16, # the dimension of the low-rank matrices
            lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout = 0.05, # dropout probability of the LoRA layers
            bias = 'none', # wether to train bias weights, set to 'none' for attention layers
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer

# Optimizer parameters here
def accelerator_parameters(model, train_dataloader, val_dataloader):
    model.model.visual.eval()
    model.model.decoder.train()
    model.project_down.train()
    model.project_up.train()
    model.model.visual.requires_grad_(False)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=logging_dir,
    )

    # fsdp_plugin = FullyShardedDataParallelPlugin()
    # accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer_parameters = filter(lambda p: p.requires_grad, model.parameters()) 
    optimizer = optimizer_cls(
        optimizer_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,)
    
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("image2image", config=vars(args))

    return accelerator, model, optimizer, train_dataloader, val_dataloader, lr_scheduler

# Train model here
def train_model(accelerator, model, optimizer, train_dataloader, val_dataloader, lr_scheduler, tokenizer):
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    transform = TF.Compose([
            TF.Resize((EVA_IMAGE_SIZE, EVA_IMAGE_SIZE), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ])

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            # dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            num_update_steps_per_epoch = math.ceil(
                    len(train_dataloader) / args.gradient_accumulation_steps
                )
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("epoch")

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        if accelerator.is_main_process:
            if (
                epoch % args.validation_epochs == 0 and args.val
            ):
                logger.info(f"Running validation... \n ")

                pipeline = DiffusionPipeline.from_pretrained(
                        'ckpts/Emu2-Gen',
                        custom_pipeline="pipeline_emu2_gen",
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        variant="bf16",
                        tokenizer=tokenizer,
                        multimodal_encoder=None,
                    )
                
                pipeline.to("cuda")

                edited_images = []
                texts = []
                with torch.autocast(
                    'cuda',
                    enabled=accelerator.mixed_precision == "fp16",
                ):
                    with torch.no_grad():
                        val_model = accelerator.unwrap_model(model)
                        val_model.eval()
                        for batch in (train_dataloader):
                            for bn in tqdm(range(len(batch['text'][:10])), desc="Generating train images"):
                                text_prompt, image_prompt = "", []
                                for x in batch['interleave_sequence_val'][bn]:
                                    if isinstance(x, str):
                                        text_prompt += x
                                    else:
                                        text_prompt += DEFAULT_IMG_PLACEHOLDER
                                        image_prompt.append(transform(x))

                                if len(image_prompt) == 0:
                                    image_prompt = None
                                else:
                                    image_prompt = torch.stack(image_prompt)
                                    image_prompt = image_prompt.to(device=accelerator.device, dtype=weight_dtype)
                                negative_prompt = {}
                                prompt = val_model.generate_image(text=[text_prompt], image=image_prompt, tokenizer=tokenizer)

                                key = ""
                                if key not in negative_prompt:
                                    negative_prompt[key] = val_model.generate_image(text=[key], tokenizer=tokenizer)
                                prompt_embeds = torch.cat([prompt, negative_prompt[key]], dim=0)
                                # prompt_embeds = val_model.generate_images([text_prompt], tokenizer, image_prompt)
                                edited_image = pipeline.validation_forward(prompt_embeds=prompt_embeds).image

                                h_concat = PIL.Image.new('RGB', (edited_image.width * 2, edited_image.height))
                                h_concat.paste(edited_image, (0, 0))
                                h_concat.paste(batch['image'][bn].resize((args.resolution, args.resolution)), (edited_image.width, 0))
                                edited_images.append(h_concat)
                                texts.append(batch['text'][bn])
                            break

                #  Log images to disk
                for img, prompt in zip(edited_images, texts):
                    os.makedirs(os.path.join(args.output_dir, 'vis', f'train_epoch[{epoch}]_step[{global_step}]'), exist_ok=True)
                    img.save(os.path.join(args.output_dir, 'vis', f'train_epoch[{epoch}]_step[{global_step}]', f"{prompt.replace(' ', '_')[:-1]}.png"))


                edited_images = []
                texts = []
                with torch.autocast(
                    'cuda',
                    enabled=accelerator.mixed_precision == "fp16",
                ):
                    for batch in (val_dataloader):
                        with torch.no_grad():
                            for bn in tqdm(range(len(batch['text'][:10])), desc="Generating val images"):

                                text_prompt, image_prompt = "", []
                                for x in batch['interleave_sequence_val'][bn]:
                                    if isinstance(x, str):
                                        text_prompt += x
                                    else:
                                        text_prompt += DEFAULT_IMG_PLACEHOLDER
                                        image_prompt.append(transform(x))

                                if len(image_prompt) == 0:
                                    image_prompt = None
                                else:
                                    image_prompt = torch.stack(image_prompt)
                                    image_prompt = image_prompt.to(device=accelerator.device, dtype=weight_dtype)
                                negative_prompt = {}
                                prompt = val_model.generate_image(text=[text_prompt], image=image_prompt, tokenizer=tokenizer)

                                key = ""
                                if key not in negative_prompt:
                                    negative_prompt[key] = val_model.generate_image(text=[key], tokenizer=tokenizer)
                                prompt_embeds = torch.cat([prompt, negative_prompt[key]], dim=0)
                                # prompt_embeds = val_model.generate_images([text_prompt], tokenizer, image_prompt)
                                edited_image = pipeline.validation_forward(prompt_embeds=prompt_embeds).image

                                h_concat = PIL.Image.new('RGB', (edited_image.width * 2, edited_image.height))
                                h_concat.paste(edited_image, (0, 0))
                                h_concat.paste(batch['image'][bn].resize((args.resolution, args.resolution)), (edited_image.width, 0))
                                edited_images.append(h_concat)
                                texts.append(batch['text'][bn])
                            break
                #  Log images to disk
                for img, prompt in zip(edited_images, texts):
                    os.makedirs(os.path.join(args.output_dir, 'vis', f'val_epoch[{epoch}]_step[{global_step}]'), exist_ok=True)
                    img.save(os.path.join(args.output_dir, 'vis', f'val_epoch[{epoch}]_step[{global_step}]', f"{prompt.replace(' ', '_')[:-1]}.png"))            

                del pipeline
                del val_model
                torch.cuda.empty_cache()

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(model):
                batch_text_prompt = []
                batch_image_prompt = []
                for inputs in batch['interleave_sequence']:
                    text_prompt, image_prompt = "", []
                    for x in inputs:
                        if isinstance(x, str):
                            text_prompt += x
                        else:
                            text_prompt += DEFAULT_IMG_PLACEHOLDER
                            image_prompt.append(transform(x))

                    if len(image_prompt) == 0:
                        image_prompt = None
                    else:
                        image_prompt = torch.stack(image_prompt)
                        image_prompt = image_prompt.to(device=accelerator.device, dtype=weight_dtype)

                    batch_text_prompt.append(text_prompt)
                    batch_image_prompt.append(image_prompt)

                loss = model(text=batch_text_prompt, image=batch_image_prompt, tokenizer=tokenizer)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    unwrap_model = accelerator.unwrap_model(model)
                    clip_param = filter(lambda p: p.requires_grad, unwrap_model.parameters())
                    accelerator.clip_grad_norm_(clip_param, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    'step': global_step,
                    "epoch": epoch
                }
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

        if epoch % args.checkpointing_steps == 0:
            if accelerator.is_main_process:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}-{epoch}"
                )
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)

        pipeline = DiffusionPipeline.from_pretrained(
            'ckpts/Emu2-Gen',
            custom_pipeline="pipeline_emu2_gen",
            torch_dtype=torch.bfloat16,
            tokenizer=tokenizer,
            use_safetensors=True,
            variant="bf16",
            multimodal_encoder=model,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()

# Main function here
if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = build_model(args)
    train_dataloader, val_dataloader = build_dataloader(args, tokenizer)
    accelerator, model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator_parameters(model, train_dataloader, val_dataloader)
    train_model(accelerator, model, optimizer, train_dataloader, val_dataloader, lr_scheduler, tokenizer)
