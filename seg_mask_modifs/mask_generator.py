import cv2
import json
import numpy as np
import torch

from PIL import Image
from torchvision.transforms import transforms as transforms

from model_utils.model_celebmask import BiSeNet


class mask_generator:
    """ Class to generate masks using models"""

    def __init__(self, threshold=0.5, auto_init=True):
        """
        Arguments:

            threshold: Minimum required model threshold on inferencing.
            auto_init: Auto initialize models whenever their label is seen.
            Initializes from default path model stored to in download_models(). If path changed then initialize manually.
        """
        self.model_preference = ['deeplabv3', 'maskrcnn', 'face']
        self.all_models = ['deeplabv3', 'maskrcnn', 'face']
        self.maskrcnn_model = False
        self.deeplabv3_model = False
        self.face_model = False
        self.threshold = threshold
        self.auto_init = auto_init
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.gpu = torch.cuda.is_available()
        self.model_labels = json.load(open('model_utils/labels.json'))

        self.label_mapping = {'deeplabv3': 'deeplab_pascal_labels',
                              'maskrcnn': 'maskrcnn_coco_labels',
                              'face': 'face_labels'}

    def init_maskrcnn(self, model_path='models/maskrcnn_resnet50_fpn.pt'):
        """ Function to initialize maskrcnn model

        Arguments:

            model_path: Path to maskrcnn model
        """
        print('Initializing maskrcnn model')
        self.maskrcnn_model = torch.load(model_path)
        self.maskrcnn_model.to(self.device).eval()
        print('Initialized maskrcnn model')

    def init_deeplabv3(self, model_path='models/deeplab_restnet101.pt'):
        """ Function to initialize deeplabv3 model

        Arguments:

            model_path: Path to deeplabv3 model
        """
        print('Initializing deeplabv3 model')
        self.deeplabv3_model = torch.load(model_path)
        self.deeplabv3_model.to(self.device).eval()
        print('Initialized deeplabv3 model')

    def init_face(self, model_path='models/face.pth'):
        """ Function to initialize face model

        Arguments:

            model_path: Path to face model
        """
        print('Initializing face model')
        self.face_model = BiSeNet(n_classes=19)
        if self.gpu:
            self.face_model.cuda()
            self.face_model.load_state_dict(torch.load(model_path))
        else:
            self.face_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.face_model.eval()
        print('Face model initialized')

    def print_model_preference(self):
        """ Function to know the current model preference"""
        print(self.model_preference)

    def set_model_preference(self, model=None, pos=0, model_list=None):
        """ Function to set the model preference.
        Only the models in model list will be used for mask generation.
        Pass either model along with its position or complete list containing models.
        If both are passed model_list will be preffered.

        Arguments:

            model: str having of 'deeplabv3', 'maskrcnn', 'face' whose value need to be set.
            pos: int having the position of preference of the model starting from 0. Default: 0.
            model_list: List containing models
        """

        if model_list is None and model is None:
            raise AttributeError("One of model or model_list needs to be passed")

        if model_list is not None:
            for models in model_list:
                if models not in self.all_models:
                    print("Wrong model name passed:", models)
                    raise ValueError
            self.model_preference = model_list
            return

        if model not in self.all_models:
            print("Wrong model name passed:", model)
            raise ValueError

        if model in self.model_preference:
            self.model_preference.remove(model)
        self.model_preference.insert(pos, model)

    def maskrcnn_inference(self, img, labels):
        """Function to perform inference using MaskRCNN

        Arguments:

            img: input image
            labels: labels to generate mask of.

        Returns:

        output_mask: A combined mask of the labels provided"""

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = transform(img_rgb)
        img_rgb = img_rgb.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.maskrcnn_model(img_rgb)

        # correct from code in docker
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        if len(outputs[0]['masks']) == 1:
            masks = (outputs[0]['masks'] > 0.5)[0].detach().cpu().numpy()
        else:
            masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_labels = [self.model_labels['maskrcnn_coco_labels'][i] for i in outputs[0]['labels']]
        output_mask = np.zeros((img.shape[:2]), dtype=np.uint8)

        for i in range(len(scores)):
            if scores[i] > self.threshold and pred_labels[i] in labels:
                output_mask = cv2.bitwise_or(output_mask, np.array(masks[i], dtype=np.uint8))

        return output_mask

    def deeplabv3_inference(self, img, labels):
        """Function to perform inference using deeplabv3

        Arguments:

            img: input image
            labels: labels to generate mask of.

        Returns:

            output_mask: A combined mask of the labels provided"""
        trf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                                  ])
        img_pil = Image.fromarray(img)
        inp = trf(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.deeplabv3_model(inp)['out']
        output_mask = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        pred_labels_ind = np.unique(output_mask)
        req_label_ind = [self.model_labels['deeplab_pascal_labels'].index(
            label) for label in labels]

        for pred_label_ind in pred_labels_ind:
            if pred_label_ind not in req_label_ind:
                # converting all unwanted labels to 0
                output_mask[output_mask[:] == pred_label_ind] = 0

        output_mask = np.array(output_mask, dtype=np.uint8)
        _, output_mask = cv2.threshold(output_mask, 0, 255, cv2.THRESH_BINARY)
        return output_mask

    def face_inference(self, img, labels):
        """Function to perform inference using face model

        Arguments:

            img: input image
            labels: labels to generate mask of.

        Returns:

            output_mask: A combined mask of the labels provided"""

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = transform(img_rgb)
        img_rgb = img_rgb.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.face_model(img_rgb)[0]
        output_mask = out.squeeze(0).cpu().numpy().argmax(0)
        output_mask = output_mask.astype(np.uint8)
        pred_labels_ind = np.unique(output_mask)
        req_label_ind = [self.model_labels['face_labels'].index(label) for label in labels]
        if "face" in labels:
            # for complete face blurring adding all label indices except hat and cloth
            # and removing its index from list
            req_label_ind.extend([i for i in range(16)])
            req_label_ind.append(17)
            req_label_ind.remove(19)
            req_label_ind = list(set(req_label_ind))

        for pred_label_ind in pred_labels_ind:
            if pred_label_ind not in req_label_ind:
                # converting all unwanted labels to 0
                output_mask[output_mask[:] == pred_label_ind] = 0

        _, output_mask = cv2.threshold(output_mask, 0, 255, cv2.THRESH_BINARY)
        return output_mask

    def generate(self, img, labels, use_model=None):
        """ Function to generate masks for labels.

        Arguments:

            img: Input image read by OpenCV
            labels: list containing one or more labels whose masks needs to be generated.
            use_model:  One of deeplabv3, maskrcnn or face.
            If argument passed only that model will be used for mask creation.

        Returns: mask of those labels.
        """

        if use_model is not None:
            if use_model not in self.all_models:
                print("use_model should be one of:", *self.all_models)
                raise ValueError

            model_labels = self.model_labels[self.label_mapping[use_model]]
            labels_skipped = list(set(labels) - set(model_labels))
            if not len(labels_skipped):
                print("Skipping labels:", *labels_skipped,
                      ", not present in labels of model:", use_model)
            labels = list(set(labels).intersection(set(model_labels)))

            # initialize model if auto_init is True and model is not initialized yet.
            if self.auto_init and not eval("self." + use_model + "_model"):
                eval("self.init_" + use_model)
            # using eval("self.__" + use_model + "_inference") to generate inference funtion
            output_mask = eval("self." + use_model + "_inference")(img, labels)
            output_mask = cv2.threshold(output_mask, 0, 255, cv2.THRESH_BINARY)
            return output_mask

        output_mask = np.zeros((img.shape[:2]), dtype=np.uint8)
        for model in self.model_preference:
            model_labels = self.model_labels[self.label_mapping[model]]
            labels_pass = list(set(labels).intersection(set(model_labels)))
            if len(labels_pass):
                # initialize model if auto_init is True and model is not initialized yet.
                if self.auto_init and not eval("self." + model + "_model"):
                    eval("self.init_" + model)()
                print("Inferencing labels:", *labels_pass, "with model:", model)
                mask = eval("self." + model + "_inference")(img, labels_pass)
                labels = list(set(labels) - set(labels_pass))
                output_mask = cv2.bitwise_or(output_mask, mask)

        if len(labels):
            print("Labels skipped:", *labels, ", not present in labels of any model")
        _, output_mask = cv2.threshold(output_mask, 0, 255, cv2.THRESH_BINARY)
        return output_mask


# Testing
if __name__ == "__main__":
    obj = mask_generator()
    img = cv2.imread('images/city.jpg')
    mask = obj.generate(img, ["person", "suitcase"])
    # print(mask)
    # mask = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('images/city_mask.jpg', mask)