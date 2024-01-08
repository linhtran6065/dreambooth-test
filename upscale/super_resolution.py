import torch
import PIL
import numpy as np
from upscale.RealESRGAN import RealESRGAN
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

class Upscaler:
    def __init__(self, scale=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale = scale
        self.model = RealESRGAN(self.device, scale=self.scale)
        self.model.load_weights(f'weights/RealESRGAN_x{self.scale}.pth', download=True)
        
    def upscale(self, img):
        # torch.cuda.empty_cache()
        upscaled_img = self.model.predict(img)
        return upscaled_img
        