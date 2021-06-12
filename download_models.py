import os

class download_models:
    """Helper class to download models"""

    def maskrcnn_model(self, save_path='models/maskrcnn_restnet50_fpn.pt'):
        """ Download and save maskrcnn_model

        Arguments:
        save_path: Path to save maskrcnn model. Must end with '.pt'. Default: 'models/maskrcnn_restnet50_fpn.pt'
        """

        import torch
        import torchvision

        if save_path[-3:] != '.pt':
            raise ValueError('Save path should end with .pt')

        if save_path == 'models/maskrcnn_restnet50_fpn.pt' and not os.path.exists('models'):
            os.makedirs('models')

        # getting base model from pytorch torchvision
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        torch.save(model, 'models/maskrcnn_resnet50_fpn.pt')

    def face_model(self, save_path='models/face.pth'):
        """ Download and save face model

        Arguments:
        save_path: Path to save maskrcnn model. Must end with '.pth'. Default: 'models/maskrcnn_restnet50_fpn.pt'
        """

        from google_drive_downloader import GoogleDriveDownloader as gdd

        if save_path[-3:] != '.pth':
            raise ValueError('Save path should end with .pth')

        if save_path == 'models/face.pth' and not os.path.exists('models'):
            os.makedirs('models')

        gdd.download_file_from_google_drive(file_id='154JgKpzCPW82qINcVieuPH3fZ2e0P812',
                                            dest_path='models/face.pth')
