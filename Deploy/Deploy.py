from Module_Detect import PaddleOCR
from Module_Recognize import VietOCR
from fastapi import FastAPI, File, UploadFile

save_dir = "Image_Cropped"
folder_img = "Image_Cropped"

#################### test #######################
# paddleocr_model = PaddleOCR(img_path, save_dir)
# paddleocr_model.run()
# vietocr_model = VietOCR(folder_img)
# results = vietocr_model.run()
# print(results)

app = FastAPI()


@app.post("/upload_image/")
async def upload_image(file_img: UploadFile = File(...)):
    with open(file_img.filename, "wb") as buffer:
        buffer.write(file_img.file.read())

    paddleocr_model = PaddleOCR(file_img.filename, save_dir)
    paddleocr_model.run()
    vietocr_model = VietOCR(folder_img)
    results = vietocr_model.run()

    return {"RESULTS": results}
