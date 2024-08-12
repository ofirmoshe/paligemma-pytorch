import torch

from modules.siglip import SigLIPConfig, SigLIPVisionTransformer
config = SigLIPConfig()
model = SigLIPVisionTransformer(config)
image = torch.rand((1,3,224,224))
print(image.shape)
output = model(image)
print(output.shape)