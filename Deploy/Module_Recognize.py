import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image


class VietOCR:
    def __init__(self, folder_img):
        self.folder_img = folder_img
        self.config = Cfg.load_config_from_name("vgg_transformer")
        self.config["device"] = 'cpu'
        self.config["weight"] = "weight/transformerocr.pth"
        self.detector = Predictor(self.config)

    def recognizer(self):
        list_texts = []
        for img_path in sorted(os.listdir(self.folder_img)):
            img = Image.open(os.path.join(self.folder_img, img_path))
            text = self.detector.predict(img)
            list_texts.append(text)

        dict_texts = {list_texts[i]: list_texts[i + 1] for i in range(0, len(list_texts), 2)}
        return dict_texts

    def run(self):
        # self.config()
        dict_results = self.recognizer()
        return dict_results


if __name__ == "__main__":
    # config = Cfg.load_config_from_name("vgg_transformer")
    # config["device"] = 'cpu'
    # detector = Predictor(config)
    #
    # img = Image.open("Image_Cropped/000000.jpg")
    # text = detector.predict(img)
    # print(text)

    folder_img = "Image_Cropped"
    model = VietOCR(folder_img)
    dict_results = model.run()
    print(dict_results)
