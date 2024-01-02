import os
from PIL import Image
from autocrop import Cropper

class FaceCropper():
    def __init__(self):
        self.cropper = Cropper()

    def crop(self, image_folder):
        cropped_count = 0
        num_images = len(os.listdir(image_folder))
        for _, filename in enumerate(os.listdir(image_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                cropped_array = self.cropper.crop(f"{image_folder}/{filename}")
                try:
                    cropped_image = Image.fromarray(cropped_array)
                    cropped_image.save(f"{image_folder}/cropped_{filename}")
                    cropped_count+=1
                except:
                    continue
            if cropped_count == num_images//2:
                break
        return "Cropping completed!"

class Inpainter():
    def __init__(self):
        pass
    
    def inpaint(self, image_folder):
        pass


# image_cropper = FaceCropper()
# image_cropper.crop("image_folder")