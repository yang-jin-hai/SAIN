from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class REALCOMP:
    def __init__(self, format='JPEG', quality=75):
        self.format = format
        self.quality = quality

    def __call__(self, tensors, out_type=np.uint8, min_max=(0, 1)):
        results = []
        for tensor in tensors:
            tensor = tensor.squeeze().float().cpu().detach().clamp_(*min_max) # clamp
            img = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            if out_type == np.uint8:
                img = (img.numpy() * 255.0).round()
            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            img = Image.fromarray(img.astype(out_type)[:,:,::-1])

            with BytesIO() as f:
                img.save(f, format=self.format, quality=self.quality)
                f.seek(0)
                img_jpg = Image.open(f)
                img_jpg.load()
            results.append(transforms.ToTensor()(img_jpg))

        return torch.stack(results, 0).to(torch.device('cuda'))
