from .model import Classifier
import torch
import numpy as np
from torchvision import transforms
import cv2 as cv

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_STATE_DICT_KEY = "model_state_dict"
CLASSES_KEY = "class_names"


class NumberClassifier:

    def __init__(self):
        self.model = Classifier().to(device=DEVICE)
        self.classes_names = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        pass
    #TODO load model

    def load_model(self, path="../models/model_with_classes.pth")->bool:
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint[MODEL_STATE_DICT_KEY])
        self.classes_names = checkpoint[CLASSES_KEY]


    def predict(self, image: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            image = self.preprocess(image)
            image = image.to(DEVICE)

            pred = self.model(image)
            pred_probs = torch.softmax(pred, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)
            class_name = self.classes_names[pred_label.cpu()]
            pred_probs = pred_probs.cpu().numpy().squeeze()
            results = [ {"class_name": name, "pred": pred_prob} for name, pred_prob in zip(self.classes_names, pred_probs)]

            return class_name, results
    
    def preprocess(self, image):
        image = cv.resize(image, IMAGE_SIZE)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = 255 - image
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor                

    #TODO add get info with % value 