import os

from PIL import Image
from autocrop import Cropper

import torch
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as T
from PIL import Image
import random
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline


class FaceCropper():
    def __init__(self):
        self.cropper = Cropper()

    def crop(self, image_folder):
        cropped_count = 0
        cropped_images_list = []
        num_images = len(os.listdir(image_folder))
        for _, filename in enumerate(os.listdir(image_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                cropped_array = self.cropper.crop(f"{image_folder}/{filename}")
                try:
                    cropped_image = Image.fromarray(cropped_array)
                    cropped_images_list.append(cropped_image)
                    cropped_image.save(f"{image_folder}/cropped_{filename}")
                    cropped_count+=1
                except:
                    continue
            if cropped_count == num_images//2:
                break
        return cropped_images_list

class Inpainter():
    def __init__(self):
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            cache_dir="stable-diffusion-2-inpainting-cache",
            torch_dtype=torch.float16,).to("cuda")

    @staticmethod
    def prepare_images(image, augmentation_scale = (0.6, 1.0)):
        mask = torch.zeros((1, 3, 512, 512)).cuda()
        mask[:, :, 30:460, 100:400] += 1
        train_transforms = T.Compose(
            [
                T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(512),
                T.RandomHorizontalFlip(),
            ]
        )
        image = ToTensor()(train_transforms(image.convert('RGB'))).unsqueeze(0).cuda() + 1e-5
        image *= mask
        image = T.RandomAffine(degrees=(-10, 10), translate=(0.2, 0.2), scale=augmentation_scale)(image)
        mask_image = 1.0 - image.clone().to(dtype=torch.bool).to(dtype=torch.int8)
        image = T.ToPILImage()(image.squeeze())
        mask_image = T.ToPILImage()(mask_image.squeeze())
        return image, mask_image
    
    def inpaint(self, gender, cropped_images_list, image_folder):
        print(cropped_images_list)
        inpainted_images = []
        prompt = f"A photo portrait of a {gender} wearing a business suit with a office background"
        negative_prompt = ""

        for index, cropped_image in enumerate(cropped_images_list):
            print(cropped_image)
            try:
                prepared_image, prepared_mask_image = self.prepare_images(image=cropped_image)
            except Exception as e:
                print(e)
                print("prepare imgs not passed")

            try:
                inpainted_image = self.inpaint_pipe(prompt=prompt, num_inference_steps=50, image=prepared_image, mask_image=prepared_mask_image, negative_prompt=negative_prompt, guidance_scale=7.5).images[0]
            except Exception as e:
                print(e)
                print("infer inpaint not passed")               
            inpainted_images.append(inpainted_image)
            inpainted_image.save(f"{image_folder}/inpainted_img_{index}.jpg")

        return inpainted_images

