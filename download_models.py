import torch
import torchvision
from google_drive_downloader import GoogleDriveDownloader as gdd

def maskrcnn_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    torch.save(model, 'models/maskrcnn_resnet50_fpn.pt')

def face_parsing_model():
    gdd.download_file_from_google_drive(file_id='154JgKpzCPW82qINcVieuPH3fZ2e0P812',
                                        dest_path='models/face_parsing.pth')


if __name__=="__main__":
    # maskrcnn_model()
    face_parsing_model()