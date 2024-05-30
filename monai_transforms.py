# Adapted from transforms.py from Contrastive Learning Framework

# The transforms used to augment the images.
import monai.transforms as M
from torchvision import transforms as T

resizeCrop = M.ResizeWithPadOrCrop((50, 50))
randWeightCrop = M.RandWeightedCrop((50,50))
randSpatialCrop = M.RandSpatialCrop(roi_size=(50, 50, 50), random_size=False)
randRotate = M.RandRotate(range_x = 0.5, prob = 1.0)
normalize = M.NormalizeIntensity()
ensureChannel = M.EnsureChannelFirst() # Adds an extra channel dimension, needed since Monai's load_image
torchNormalize = T.ToTensor()
resize = M.Resize((50, 50, 50))

composedTransform = M.Compose([randSpatialCrop, randRotate])
identityTransform = M.Compose([resize])

