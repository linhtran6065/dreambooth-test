# # import argparse
# # from argparse import Namespace

# # def convert_namespace_to_parser(args):
# #     parser = argparse.ArgumentParser(description="Recreate argument parser from Namespace object.")
# #     for arg in vars(args):
# #         value = getattr(args, arg)
# #         arg_type = type(value)
# #         parser.add_argument(f'--{arg}', type=arg_type, default=value)

# #     return parser

# # args = {
# #     "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
# #     "pretrained_vae_name_or_path": "stabilityai/sd-vae-ft-mse",
# #     "push_to_hub": False,
# #     "hub_token": None,
# #     "hub_model_id": None,
# #     "save_interval": 10000,  # not used
# #     "save_min_steps": 0,
# #     "mixed_precision": "fp16",
# #     "not_cache_latents": False,
# #     "local_rank": -1,
# #     "concepts_list": None,
# #     "logging_dir": "logs",
# #     "log_interval": 10,
# #     "hflip": False,
# # }
# # args = Namespace(**args)
# # # print(args)

# # # Example usage
# # new_parser = convert_namespace_to_parser(args)

# # args = new_parser.parse_args()
# # print(args)


# import os
# from zipfile import ZipFile
# import mimetypes

# instance_data = "test.zip"
# instance_dir_name = f"train_man"
# instance_subdir_name = f"1_ohwx man"
# output_dir = "checkpoints"

# # Create main directories if they don't exist
# for dir_name in [instance_dir_name, output_dir]:
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)

# # Extract instance_data ZIP file
# instance_data_path = os.path.join(instance_dir_name, instance_subdir_name)

# if not os.path.exists(instance_data_path):
#     os.makedirs(instance_data_path)


# with ZipFile(str(instance_data), "r") as zip_ref:
#         for zip_info in zip_ref.infolist():
#             if not zip_info.filename.endswith("/") and not zip_info.filename.startswith("__MACOSX"):
#                 mt = mimetypes.guess_type(zip_info.filename)
#                 if mt and mt[0] and mt[0].startswith("image/"):
#                     zip_info.filename = os.path.join(instance_subdir_name, os.path.basename(zip_info.filename))
#                     zip_ref.extract(zip_info, instance_dir_name)

import json

def remove_duplicates(lst):
    return list(set(lst))

with open('prompt_data.json', 'r') as file:
    data = json.load(file)

print(len(data["<object>"]))
new_list = remove_duplicates(data["<object>"])

print(new_list)
print(len(new_list))
