import os
import torch
import torchvision
from google_drive_downloader import GoogleDriveDownloader as gdd

def maskrcnn_coco(save_path='models/maskrcnn_restnet50_fpn.pt'):
    """ Download and save maskrcnn model

    Arguments:
    save_path: Path to save maskrcnn model. Must end with '.pt'. Default: 'models/maskrcnn_restnet50_fpn.pt'
    """

    if save_path[-3:] != '.pt':
        raise ValueError('Save path should end with .pt')

    if save_path == 'models/maskrcnn_restnet50_fpn.pt' and not os.path.exists('models'):
        os.makedirs('models')

    # getting base model from pytorch torchvision
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    torch.save(model, 'models/maskrcnn_resnet50_fpn.pt')

def deeplab_pascal(save_path='models/deeplab_restnet101.pt'):
    """ Download and save deeplab model

    Arguments:
    save_path: Path to save deeplab model. Must end with '.pt'. Default: 'models/deeplab_restnet101.pt'
    """

    if save_path[-3:] != '.pt':
        raise ValueError('Save path should end with .pt')

    if save_path == 'models/deeplab_restnet101.pt' and not os.path.exists('models'):
        os.makedirs('models')

    # getting base model from pytorch torchvision
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    torch.save(model, 'models/deeplab_restnet101.pt')

def face(save_path='models/face.pth'):
    """ Download and save face model

    Arguments:
    save_path: Path to save face model. Must end with '.pth'. Default: 'models/face.pth'
    """

    if save_path[-3:] != '.pth':
        raise ValueError('Save path should end with .pth')

    if save_path == 'models/face.pth' and not os.path.exists('models'):
        os.makedirs('models')

    gdd.download_file_from_google_drive(file_id='154JgKpzCPW82qINcVieuPH3fZ2e0P812',
                                        dest_path='models/face.pth')