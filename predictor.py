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

    # parser.add_argument(
    #     "--learning_rate_te",
    #     type=float,
    #     default=None,
    #     help="learning rate for text encoder, default is same as unet / Text Encoderの学習率、デフォルトはunetと同じ",
    # )
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

        gender: str = Input(
            description="gender",
        ),

        identifier: str = Input(
            description="identifier",
            default="ohwx"
        ),

        num_repeats: str = Input(
            description="num_repeats",
            default="1"
        ), 

        output_name: str = Input(
            description="model output name",
        ),

        instance_data: Path = Input(
            description="A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your 'subject' that you want the trained model to embed in the output domain for later generating customized scenes beyond the training images. For best results, use images without noise or unrelated objects in the background.",
        ),
        class_data: Path = Input(
            description="An optional ZIP file containing the training data of class images. This corresponds to `class_prompt` above, also with the purpose of keeping the model generalizable. By default, the pretrained stable-diffusion model will generate N images (determined by the `num_class_images` you set) based on the `class_prompt` provided. But to save time or to have your preferred specific set of `class_data`, you can also provide them in a ZIP file.",
            default=None,
        ),

        train_batch_size: int = Input(
            description="Batch size (per device) for the training dataloader.",
            default=1,
        ),

        max_train_steps: int = Input(
            description="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
            default=2800,
        ),

        save_every_n_epochs: int = Input(
            description="Total number of epochs to save",
            default=3000,
        ),

        learning_rate: float = Input(
            description="Initial learning rate (after the potential warmup period) to use.",
            default=1e-6,
        ),

        learning_rate_te: float = Input(
            description="Initial learning rate te (after the potential warmup period) to use.",
            default=1e-6,
        ),


        stop_text_encoder_training: float = Input(
            description="stop_text_encoder_training ratio",
            default=0.8,
        ),

        noise_offset: float = Input(
            description="noise_offset",
            default=0.0,
        ),

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

    ) -> Path:

        instance_dir_name = f"train_{gender}"
        instance_subdir_name = f"{num_repeats}_{identifier} {gender}"

        class_dir_name = f"reg_{gender}"
        class_subdir_name = f"{num_repeats}_{gender}"

        output_dir = "checkpoints"

        try:
            # Create main directories if they don't exist
            for dir_name in [instance_dir_name, class_dir_name, output_dir]:
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

            # Extract instance_data ZIP file
            instance_data_path = os.path.join(instance_dir_name, instance_subdir_name)
            class_data_path = os.path.join(class_dir_name, class_subdir_name)

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
        except:
            print("create dir not passed")

        # some settings are fixed for the replicate model
        args = {
            "pretrained_model_name_or_path": "stablediffusionapi/realistic-vision-51",
            "train_data_dir": instance_dir_name,
            "reg_data_dir": class_dir_name,
            "train_batch_size": train_batch_size,
            "max_train_steps": max_train_steps,
            "save_every_n_epochs": save_every_n_epochs,
            "learning_rate": learning_rate,
            "learning_rate_te": learning_rate_te,
            "stop_text_encoder_training": stop_text_encoder_training,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "mixed_precision": "fp16",
            "save_precision": "fp16",
            "optimizer_type": "AdamW8bit",
            "max_data_loader_n_workers": 0,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_model_as":"safetensors",
            "enable_bucket": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": 2048,
            "bucket_reso_steps": 64,
            "cache_latents": True,
            # "cache_latents_to_disk": True,
            "xformers": True,
            "bucket_no_upscale": True,
            "noise_offset": noise_offset,
            "adaptive_noise_scale": True,
            # "scale_v_pred_loss_like_noise_pred":True,
            "await-explicit-shutdown":True,
            "num_cpu_threads_per_process": 2,
            "upload-url":"http://api.tenant-replicate-prdsvcs.svc.cluster.local/_internal/file-upload/"
        }

        try:
            args = Namespace(**args)
            print(args)
            try:
                parser = convert_namespace_to_parser(args)
            except:
                print("convert_namespace_to_parser not passed")
            try:
                parser = setup_parser(parser)
            except:
                print("setup_parser not passed")
            try:
                args = parser.parse_args()
            except:
                print("parse_args not passed")
            try:
                args = train_util.read_config_from_file(args, parser)
            except:
                print("read_config_from_file not passed")        
        except:
            print("make args not passed")

        try:
            main(args)
        except Exception as e:
            print(e)
            print("main not passed")

        gc.collect()
        torch.cuda.empty_cache()
        call("nvidia-smi")

        out_path = "output.zip"

        try:
            directory = Path(output_dir)
            with ZipFile(out_path, "w") as zip:
                for file_path in directory.rglob("*"):
                    print(file_path)
                    zip.write(file_path, arcname=file_path.relative_to(directory))
        except:
            print("zip file not passed")

        return Path(out_path)