from torchvision.transforms import transforms
from .gaussian_blur import GaussianBlur

def get_simclr_data_transforms(input_shape, mean, std, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=input_shape[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * input_shape[0])),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean=mean, std=std)])
    return data_transforms