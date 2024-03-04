import paddleocr
import cv2
from PIL import Image
import os
import numpy as np

np.int = np.int32


class PaddleOCR:
    def __init__(self, img_path, save_dir):
        self.img_path = img_path
        self.save_dir = save_dir
        self.model = paddleocr.PaddleOCR(use_angle_cls=True, use_gpu=False, lang="en")

    def detect_text(self):
        image = cv2.imread(self.img_path)
        results = self.model.ocr(image, cls=True)
        bboxes = [line[0] for line in results]
        return bboxes

    def split_bb(self, bboxes):
        os.makedirs(self.save_dir, exist_ok=True)

        count = 0
        img = Image.open(self.img_path)
        for box in bboxes:
            img_cropped = img.crop((box[0][0], box[0][1], box[2][0], box[2][1])).convert(
                "RGB")  # điểm trái trên và phải dưới
            file_name = f"{count:06d}.jpg"

            img_cropped.save(os.path.join(self.save_dir, file_name))
            count += 1

        if len(os.listdir(self.save_dir)) != 0:
            print("FOLDER CREATED SUCCESSFULLY !!")
        else:
            print("FOLDER CREATION FAILED !!")

    def run(self):
        bboxes = self.detect_text()
        self.split_bb(bboxes)


if __name__ == "__main__":
    img_path = "Screenshot 2024-02-27 132526.png"
    save_dir = "Image_cropped"
    model = PaddleOCR(img_path, save_dir)
    model.run()
