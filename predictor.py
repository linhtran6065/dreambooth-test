import os
import gc
import mimetypes
import shutil
import tempfile
import argparse
from zipfile import ZipFile
from subprocess import call, check_call
from argparse import Namespace
import time
import torch
import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions

from cog import BasePredictor, Input, Path

from dreambooth import main


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)

def setup_parser(parser) -> argparse.ArgumentParser:
    # parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, False, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--learning_rate_te",
        type=float,
        default=None,
        help="learning rate for text encoder, default is same as unet / Text Encoderの学習率、デフォルトはunetと同じ",
    )
    parser.add_argument(
        "--no_token_padding",
        action="store_true",
        help="disable token padding (same as Diffuser's DreamBooth) / トークンのpaddingを無効にする（Diffusers版DreamBoothと同じ動作）",
    )
    parser.add_argument(
        "--stop_text_encoder_training",
        type=float,
        default=0.8,
        help="steps to stop text encoder training, -1 for no training / Text Encoderの学習を止めるステップ数、-1で最初から学習しない",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )

    return parser

def convert_namespace_to_parser(args):
    parser = argparse.ArgumentParser(description="Recreate argument parser from Namespace object.")
    for arg in vars(args):
        value = getattr(args, arg)
        arg_type = type(value)
        parser.add_argument(f'--{arg}', type=arg_type, default=value)

    return parser

class Predictor(BasePredictor):
    def setup(self):
        # HACK: wait a little bit for instance to be ready
        time.sleep(10)
        check_call("nvidia-smi", shell=True)
        assert torch.cuda.is_available()

    def predict(
        self,

        gender: str = Input(#new
            description="gender",
        ),

        identifier: str = Input(#new
            description="identifier",
            default="ohwx"
        ),

        num_repeats: str = Input(#new
            description="num_repeats",
            default="1"
        ), 

        train_data_dir: str = Input(#new
            description="train data dir",
        ),

        reg_data_dir: str = Input(#new
            description="reg data dir",
        ),

        stop_text_encoder_training: int = Input(#new
            description="stop text encoder training step",
        ),

        output_name: str = Input(#new
            description="model output name",
        ),

        # instance_prompt: str = Input(
        #     description="The prompt you use to describe your training images, in the format: `a [identifier] [class noun]`, where the `[identifier]` should be a rare token. Relatively short sequences with 1-3 letters work the best (e.g. `sks`, `xjy`). `[class noun]` is a coarse class descriptor of the subject (e.g. cat, dog, watch, etc.). For example, your `instance_prompt` can be: `a sks dog`, or with some extra description `a photo of a sks dog`. The trained model will learn to bind a unique identifier with your specific subject in the `instance_data`.",
        # ),
        # class_prompt: str = Input(
        #     description="The prompt or description of the coarse class of your training images, in the format of `a [class noun]`, optionally with some extra description. `class_prompt` is used to alleviate overfitting to your customised images (the trained model should still keep the learnt prior so that it can still generate different dogs when the `[identifier]` is not in the prompt). Corresponding to the examples of the `instant_prompt` above, the `class_prompt` can be `a dog` or `a photo of a dog`.",
        # ),
        instance_data: Path = Input(
            description="A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your 'subject' that you want the trained model to embed in the output domain for later generating customized scenes beyond the training images. For best results, use images without noise or unrelated objects in the background.",
        ),
        class_data: Path = Input(
            description="An optional ZIP file containing the training data of class images. This corresponds to `class_prompt` above, also with the purpose of keeping the model generalizable. By default, the pretrained stable-diffusion model will generate N images (determined by the `num_class_images` you set) based on the `class_prompt` provided. But to save time or to have your preferred specific set of `class_data`, you can also provide them in a ZIP file.",
            default=None,
        ),
        # num_class_images: int = Input(
        #     description="Minimal class images for prior preservation loss. If not enough images are provided in class_data, additional images will be"
        #     " sampled with class_prompt.",
        #     default=50,
        # ),
        # save_sample_prompt: str = Input(
        #     description="The prompt used to generate sample outputs to save.",
        #     default=None,
        # ),
        # save_sample_negative_prompt: str = Input(
        #     description="The negative prompt used to generate sample outputs to save.",
        #     default=None,
        # ),
        # n_save_sample: int = Input(
        #     description="The number of samples to save.",
        #     default=4,
        # ),
        # save_guidance_scale: float = Input(
        #     description="CFG for save sample.",
        #     default=7.5,
        # ),
        # save_infer_steps: int = Input(
        #     description="The number of inference steps for save sample.",
        #     default=50,
        # ),
        # pad_tokens: bool = Input(
        #     description="Flag to pad tokens to length 77.",
        #     default=False,
        # ),
        # with_prior_preservation: bool = Input(
        #     description="Flag to add prior preservation loss.",
        #     default=True,
        # ),
        # prior_loss_weight: float = Input(
        #     description="Weight of prior preservation loss.",
        #     default=1.0,
        # ),
        # seed: int = Input(description="A seed for reproducible training", default=1337),
        # resolution: int = Input(
        #     description="The resolution for input images. All the images in the train/validation dataset will be resized to this"
        #     " resolution.",
        #     default=512,
        # ),
        # center_crop: bool = Input(
        #     description="Whether to center crop images before resizing to resolution",
        #     default=False,
        # ),
        # train_text_encoder: bool = Input(
        #     description="Whether to train the text encoder",
        #     default=True,
        # ),
        train_batch_size: int = Input(
            description="Batch size (per device) for the training dataloader.",
            default=1,
        ),
        # sample_batch_size: int = Input(
        #     description="Batch size (per device) for sampling images.",
        #     default=4,
        # ),
        # num_train_epochs: int = Input(default=1),
        max_train_steps: int = Input(
            description="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
            default=2800,
        ),

        save_every_n_epochs: int = Input(
            description="Total number of epochs to save",
            default=3000,
        ),
        # gradient_accumulation_steps: int = Input(
        #     description="Number of updates steps to accumulate before performing a backward/update pass.",
        #     default=1,
        # ),
        # gradient_checkpointing: bool = Input(
        #     description="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
        #     default=False,
        # ),
        learning_rate: float = Input(
            description="Initial learning rate (after the potential warmup period) to use.",
            default=1e-6,
        ),

        learning_rate_te: float = Input(
            description="Initial learning rate te (after the potential warmup period) to use.",
            default=1e-6,
        ),

        # scale_lr: bool = Input(
        #     description="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
        #     default=False,
        # ),
        lr_scheduler: str = Input(
            description="The scheduler type to use",
            choices=[
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
            default="cosine",
        ),
        lr_warmup_steps: float = Input(
            description="Number of steps for the warmup in the lr scheduler.",
            default=0.1,
        ),
        # use_8bit_adam: bool = Input(
        #     description="Whether or not to use 8-bit Adam from bitsandbytes.",
        #     default=False,
        # ),
        # adam_beta1: float = Input(
        #     default=0.9,
        #     description="The beta1 parameter for the Adam optimizer.",
        # ),
        # adam_beta2: float = Input(
        #     default=0.999,
        #     description="The beta2 parameter for the Adam optimizer.",
        # ),
        # adam_weight_decay: float = Input(
        #     default=1e-2,
        #     description="Weight decay to use",
        # ),
        # adam_epsilon: float = Input(
        #     default=1e-8,
        #     description="Epsilon value for the Adam optimizer",
        # ),
        # max_grad_norm: float = Input(
        #     default=1.0,
        #     description="Max gradient norm.",
        # ),
        # save_interval: int = Input(
        #     default=10000,
        #     description="Save weights every N steps.",
        # ),
    ) -> Path:

        # cog_instance_data = "train_data"
        # cog_class_data = "reg_data"
        # cog_output_dir = "checkpoints"
        # Dynamic directory naming based on input parameters
        instance_dir_name = f"train_{gender}"
        instance_subdir_name = f"{num_repeats}_{identifier} {gender}"

        class_dir_name = f"reg_{gender}"
        class_subdir_name = f"{num_repeats}_{gender}"

        output_dir = "checkpoints"

        # Create main directories if they don't exist
        for dir_name in [instance_dir_name, class_dir_name, output_dir]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        # Extract instance_data ZIP file
        instance_data_path = os.path.join(instance_dir_name, instance_subdir_name)
        # class_data_path = os.path.join(class_dir_name, class_subdir_name)

        if not os.path.exists(instance_data_path):
            os.makedirs(instance_data_path)

        if not os.path.exists(class_data_path):
            os.makedirs(class_data_path)

        with ZipFile(str(instance_data), "r") as zip_ref:
                for zip_info in zip_ref.infolist():
                    if not zip_info.filename.endswith("/") and not zip_info.filename.startswith("__MACOSX"):
                        mt = mimetypes.guess_type(zip_info.filename)
                        if mt and mt[0] and mt[0].startswith("image/"):
                            zip_info.filename = os.path.join(instance_subdir_name, os.path.basename(zip_info.filename))
                            zip_ref.extract(zip_info, instance_dir_name)

        # Extract class_data ZIP file
        class_data_path = os.path.join(class_dir_name, class_subdir_name)
        if class_data is not None and not os.path.exists(class_data_path):
            os.makedirs(class_data_path)

            with ZipFile(str(class_data), "r") as zip_ref:
                for zip_info in zip_ref.infolist():
                    if not zip_info.filename.endswith("/") and not zip_info.filename.startswith("__MACOSX"):
                        mt = mimetypes.guess_type(zip_info.filename)
                        if mt and mt[0] and mt[0].startswith("image/"):
                            zip_info.filename = os.path.join(class_subdir_name, os.path.basename(zip_info.filename))
                            zip_ref.extract(zip_info, class_dir_name)

        # for path in [cog_instance_data, cog_output_dir, cog_class_data]:
        #     if os.path.exists(path):
        #         shutil.rmtree(path)
        #     os.makedirs(path)

        # # extract zip contents, flattening any paths present within it
        # with ZipFile(str(instance_data), "r") as zip_ref:
        #     for zip_info in zip_ref.infolist():
        #         if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
        #             "__MACOSX"
        #         ):
        #             continue
        #         mt = mimetypes.guess_type(zip_info.filename)
        #         if mt and mt[0] and mt[0].startswith("image/"):
        #             zip_info.filename = os.path.basename(zip_info.filename)
        #             zip_ref.extract(zip_info, cog_instance_data)

        # if class_data is not None:
        #     with ZipFile(str(class_data), "r") as zip_ref:
        #         for zip_info in zip_ref.infolist():
        #             if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
        #                 "__MACOSX"
        #             ):
        #                 continue
        #             mt = mimetypes.guess_type(zip_info.filename)
        #             if mt and mt[0] and mt[0].startswith("image/"):
        #                 zip_info.filename = os.path.basename(zip_info.filename)
        #                 zip_ref.extract(zip_info, cog_class_data)

        # some settings are fixed for the replicate model
        args = {
            "pretrained_model_name_or_path": "stablediffusionapi/realistic-vision-51", #keep
            # "pretrained_vae_name_or_path": "stabilityai/sd-vae-ft-mse",
            # "revision": "fp16",
            # "tokenizer_name": None,
            "train_data_dir": instance_dir_name,
            "reg_data_dir": class_dir_name,
            # "instance_prompt": instance_prompt,
            # "class_prompt": class_prompt,
            # "save_sample_prompt": save_sample_prompt,
            # "save_sample_negative_prompt": save_sample_negative_prompt,
            # "n_save_sample": n_save_sample,
            # "save_guidance_scale": save_guidance_scale,
            # "save_infer_steps": save_infer_steps,
            # "pad_tokens": pad_tokens,
            # "with_prior_preservation": with_prior_preservation,
            # "prior_loss_weight": prior_loss_weight,
            # "num_class_images": num_class_images,
            # "seed": seed,
            # "resolution": resolution,
            # "center_crop": center_crop,
            # "train_text_encoder": train_text_encoder,
            "train_batch_size": train_batch_size,
            # "sample_batch_size": sample_batch_size,
            # "num_train_epochs": num_train_epochs,
            "max_train_steps": max_train_steps,
            "save_every_n_epochs": save_every_n_epochs,
            # "gradient_accumulation_steps": gradient_accumulation_steps,
            # "gradient_checkpointing": gradient_checkpointing,
            "learning_rate": learning_rate,
            "learning_rate_te": learning_rate_te,
            # "scale_lr": scale_lr,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            # "use_8bit_adam": use_8bit_adam,
            # "adam_beta1": adam_beta1,
            # "adam_beta2": adam_beta2,
            # "adam_weight_decay": adam_weight_decay,
            # "adam_epsilon": adam_epsilon,
            # "max_grad_norm": max_grad_norm,
            # "push_to_hub": False,
            # "hub_token": None,
            # "hub_model_id": None,
            # "save_interval": 10000,  # not used
            # "save_min_steps": 0,
            "mixed_precision": "fp16", #keep
            "save_precision": "fp16", #new
            "optimizer_type": "AdamW8bit",
            "max_data_loader_n_workers": 0,
            # "not_cache_latents": False,
            # "local_rank": -1,
            "output_dir": output_dir, #keep
            "output_name": output_name,
            # "concepts_list": None,
            # "logging_dir": "logs",
            # "log_interval": 10,
            # "hflip": False,
        }

        args = Namespace(**args)

        parser = convert_namespace_to_parser(args)
        parser = setup_parser(parser)

        args = parser.parse_args()
        args = train_util.read_config_from_file(args, parser)

        main(args)

        gc.collect()
        torch.cuda.empty_cache()
        call("nvidia-smi")

        out_path = "output.zip"

        directory = Path(output_dir)
        with ZipFile(out_path, "w") as zip:
            for file_path in directory.rglob("*"):
                print(file_path)
                zip.write(file_path, arcname=file_path.relative_to(directory))

        return Path(out_path)