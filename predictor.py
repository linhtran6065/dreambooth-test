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
from augment_data import FaceCropper, Inpainter

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
    # parser.add_argument(
    #     "--stop_text_encoder_training",
    #     type=float,
    #     default=0.8,
    #     help="steps to stop text encoder training, -1 for no training / Text Encoderの学習を止めるステップ数、-1で最初から学習しない",
    # )
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
            description="Specify the gender (for example: man/woman).",
        ),

        identifier: str = Input(
            description="Unique identifier for the model instance.",
            default="ohwx"
        ),

        num_repeats: str = Input(
            description="Number of times the input images should be repeated.",
            default="1"
        ), 

        output_name: str = Input(
            description="Name of the model's output file. (for example: output_model)",
        ),

        instance_data: Path = Input(
            description="A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your 'subject' that you want the trained model to embed in the output domain for later generating customized scenes beyond the training images. For best results, use images without noise or unrelated objects in the background.",
        ),
        class_data: Path = Input(
            description="An optional ZIP file containing the training data of class images. This corresponds to `class_prompt` above, also with the purpose of keeping the model generalizable. By default, the pretrained stable-diffusion model will generate N images (determined by the `num_class_images` you set) based on the `class_prompt` provided. But to save time or to have your preferred specific set of `class_data`, you can also provide them in a ZIP file.",
            default=None,
        ),

        using_crop_images: bool = Input(
            description="Whether to use crop images or not.",
            default=True,
        ),

        using_inpainting_images: bool = Input(
            description="Whether to use inpainting images or not.",
            default=True,
        ),

        train_batch_size: int = Input(
            description="Batch size for training data loader, applied per device.",
            default=1,
        ),

        # max_train_steps: int = Input(
        #     description="Total number of training steps to perform. It should be number of input images * 100. For example, 14 images * 100 = 1400 steps)",
        #     # default=2800,
        # ),

        # save_every_n_epochs: int = Input(
        #     description="Total number of epochs to save",
        #     default=3000,
        # ),

        learning_rate: float = Input(
            description="Initial learning rate (after the potential warmup period) to use.",
            default=1e-6,
        ),

        learning_rate_te: float = Input(
            description="Initial learning rate te (after the potential warmup period) to use.",
            default=1e-6,
        ),

        # stop_text_encoder_training: float = Input(
        #     description="stop_text_encoder_training ratio",
        #     default=0.8,
        # ),

        # noise_offset: float = Input(
        #     description="noise_offset",
        #     default=0.0,
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
        # lr_warmup_steps: float = Input(
        #     description="The ratio Number of steps for the warmup in the lr scheduler.",
        #     default=0.1,
        # ),

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

            if os.path.exists(instance_data_path):
                shutil.rmtree(instance_data_path)
            else:  
                os.makedirs(instance_data_path)

            if os.path.exists(class_data_path):
                shutil.rmtree(class_data_path)
            else:
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

        num_original_images = len(os.listdir(instance_data_path))

        # Add crop images if set to True
        if using_crop_images == True:
            print("Cropping images...")
            image_cropper = FaceCropper()
            cropped_images_list = image_cropper.crop(instance_data_path)

        try:
            if using_inpainting_images == True:
                print("Inpainting images...")
                inpainter = Inpainter()
                inpainted_images_list = inpainter.inpaint(gender, cropped_images_list, instance_data_path)
        except Exception as e:
            print(e)
            print("Inpainting not passed")
        

        print(f"Number of cropped images: {len(cropped_images_list)}")
        print(f"Number of inpainted images: {len(inpainted_images_list)}")
        print(f"Number of original images: {num_original_images}")
        num_train_images = len(os.listdir(instance_data_path))
        print(f"Number of training images: {num_train_images}")

        # Calculate steps
        training_steps = 150*num_original_images+100*len(cropped_images_list)+50*len(inpainted_images_list)

        # some settings are fixed for the replicate model
        args = {
            "pretrained_model_name_or_path": "stablediffusionapi/realistic-vision-v51",
            "train_data_dir": instance_dir_name,
            "reg_data_dir": class_dir_name,
            "using_crop_images": using_crop_images,
            "train_batch_size": train_batch_size,
            "max_train_steps": training_steps,
            "save_every_n_epochs": 3000,
            "learning_rate": learning_rate,
            "learning_rate_te": learning_rate_te,
            "stop_text_encoder_training": 0.8,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": 0.1,
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
            "cache_latents_to_disk": True,
            "xformers": True,
            "bucket_no_upscale": True,
            "await-explicit-shutdown":True,
            "upload-url":"http://api.tenant-replicate-prdsvcs.svc.cluster.local/_internal/file-upload/"
        }


        args = Namespace(**args)
        print(args)

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