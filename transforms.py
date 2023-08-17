# The transforms used to augment the images.

from torchvision import transforms as T

composedTransform = T.Compose([T.ToTensor(), T.RandomResizedCrop(size=(32,32)), 
                                       T.RandomRotation((-15, 15)), T.GaussianBlur(kernel_size=5)])

identityTransform = T.Compose([T.ToTensor(), T.Resize((32, 32))])
