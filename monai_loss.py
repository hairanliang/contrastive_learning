import monai
from monai.losses import ContrastiveLoss

contrastive_loss = ContrastiveLoss(temperature=1)


