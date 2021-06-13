import torch
from model_utils.model_celebmask import BiSeNet

#TODO adding docs for init and init of models along with model.eval() and setting device.
class mask_generator:
    """ Class to generate masks using models"""
    def __init__(self, auto_init=True):
        """
        Arguments: 

        auto_init: Auto initialize models whenever their label is seen. 
        Initializes from default path model stored to in download_models(). If path changed then initialize manually.
        """
        self.model_preference = ['deeplabv3', 'maskrcnn', 'face']
        self.maskrcnn_model = False
        self.deeplab_model = False
        self.face_model = False
        self.auto_init = auto_init
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.gpu = torch.cuda.is_available()
    
    def init_maskrcnn(self, model_path='models/maskrcnn_resnet50_fpn.pt'):
        """ Function to initialize maskrcnn model

        Arguments:

        model_path: Path to maskrcnn model
        """
        self.maskrcnn_model = torch.load(model_path)
        self.maskrcnn_model.to(self.device).eval()

    def init_deeplab(self, model_path='models/deeplab_restnet101.pt'):
        """ Function to initialize deeplab model

        Arguments:

        model_path: Path to deeplab model
        """
        self.deeplab_model = torch.load(model_path)
        self.deeplab_model.to(self.device).eval()

    def init_face(self, model_path='models/face.pth'):
        """ Function to initialize face model

        Arguments:

        model_path: Path to face model
        """
        self.face_model = BiSeNet(n_classes=19)
        if self.gpu:
            self.face_model.cuda()
            self.face_model.load_state_dict(torch.load(model_path))
        else:
            self.face_model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
        self.face_model.eval()

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

        if model_list == None and model == None:
            raise AttributeError("One of model or model_list needs to be passed")

        all_models = ['deeplabv3', 'maskrcnn', 'face']

        if model_list != None:
            for models in model_list:
                if models not in all_models:
                    print("Wrong model name passed:", models)
                    raise ValueError
            self.model_preference = model_list
            return
        
        if model not in all_models:
            print("Wrong model name passed:", model)
            raise ValueError

        if model in self.model_preference:
            self.model_preference.remove(model)
        self.model_preference.insert(model, pos)
