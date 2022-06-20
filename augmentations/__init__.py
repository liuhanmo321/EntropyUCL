from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single

imagenet_mean_std = [[0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2615]]

def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None, mean_std=imagenet_mean_std):
    if train==True:
        augmentation = SimSiamTransform(image_size, mean_std)
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier, normalize=mean_std)
    else:
        raise Exception
    
    return augmentation








