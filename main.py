from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Request, status, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger
from pydantic import BaseSettings


from starlette.responses import FileResponse

import pandas as pd
import os
import glob
import sys
import json
import datetime
import numpy as np
import pandas as pd
import cv2
import json
import base64


from PIL import Image
import tensorflow as tf
from imantics import Polygons, Mask
from keras import backend as K

from mrcnn.config import Config
from mrcnn import model as modellib, utils


origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "https://centrovision-ui.herokuapp.com"
]

class Settings(BaseSettings):
    # ... The rest of our FastAPI settings

    BASE_URL = "http://localhost:8000"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"
    
settings = Settings()


def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.USE_NGROK:
    # pyngrok should only ever be installed or initialized in a dev environment when this flag is set
    from pyngrok import ngrok

    # Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
    # when starting the server
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 8000

    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(port).public_url
    logger.info("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

    # Update any base URLs or webhooks to use the public ngrok URL
    settings.BASE_URL = public_url
    init_webhooks(public_url)

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "crack"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    GPU_COUNT = 1


config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='model')
model.load_weights('model/mask_rcnn_crack_1292.h5', by_name=True)
class_names = ['BG','color_defect','crack_wall','crack_column','hole','crack_floor','rust','water_stain','white_stain']

def draw_caption(img, x1,x2,y1,y2,caption,txt_color,bg_color):
    (text_width, text_height) = cv2.getTextSize(caption,cv2.FONT_HERSHEY_COMPLEX,1,3)[0]
    ## left
    if y1 - text_height >= 70 and x1 - 40 >= 0:
        img = cv2.rectangle(img, (x1-40 if x1+text_width < img.shape[1] else img.shape[1]-text_width-20, y1-80), (x1+text_width, y1-20), bg_color,-1)
        img = cv2.putText(img, caption, (x1-20 if x1+text_width < img.shape[1] else img.shape[1]-text_width, y1-40), cv2.FONT_HERSHEY_COMPLEX, 1, txt_color, 3)
    elif y1 - text_height >= 70 and x1 - 40 < 0:
        img = cv2.rectangle(img, (x1, y1-80), (x1+text_width+10, y1-20), bg_color,-1)
        img = cv2.putText(img, caption, (x1+10, y1-40), cv2.FONT_HERSHEY_COMPLEX, 1, txt_color, 3)
    elif y1 - text_height < 70 and x1 - 40 >= 0 and img.shape[1] - x1 - text_width >= 20:
        img = cv2.rectangle(img, (x1-40, y2+10), (x1+text_width, y2+50), bg_color, -1)
        img = cv2.putText(img, caption, (x1-20, y2+40), cv2.FONT_HERSHEY_COMPLEX, 1, txt_color, 3)
    elif y1 - text_height < 70 and x1 - 40 < 0:
        img = cv2.rectangle(img, (x1, y2+10), (x1+text_width+20, y2+50), bg_color, -1)
        img = cv2.putText(img, caption, (x1+10, y2+40), cv2.FONT_HERSHEY_COMPLEX, 1, txt_color, 3)
    ## right
    elif y1 - text_height < 70 and x1 - 40 >= 0 and img.shape[1] - x1 - text_width < 20 :
        img = cv2.rectangle(img, (img.shape[1]-text_width-20, y2+10), (x1+text_width, y2+50), bg_color, -1)
        img = cv2.putText(img, caption, (img.shape[1]-text_width, y2+40), cv2.FONT_HERSHEY_COMPLEX, 1, txt_color, 3)
        
    return img  

def detect_defect(image, model=model):
    r = model.detect([np.array(image)], verbose=1)[0]
    return r

def save_upload(image, filename):
    name = f"upload/{filename}"
    cv2.imwrite(name, image)

    return True

def save_detect(image, filename):
    cv2.imwrite(f"detected/{filename}", image)
    
    
def display_instances(image, boxes, ids):
    n_instances = boxes.shape[0]
    colors = [tuple((0,0,255)) for _ in range(n_instances)]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        label = class_names[ids[i]]
        caption = f'{label}'
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        (text_width, text_height) = cv2.getTextSize(caption,cv2.FONT_HERSHEY_COMPLEX,1,3)[0]
        image = draw_caption(image, x1,x2,y1,y2,caption,color,(0,0,0))
       
    return image


def verify_token(req: Request):
    if 'authorization' in req.headers.keys():
        token = req.headers["authorization"]
        # Here your code for verifying the token or whatever you use
        if token != 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9':
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )
        return True
    else:
        raise HTTPException(
                status_code=401,
                detail="No authorize header"
            )


@app.post("/image")
async def predict_from_image(listFiles: List[UploadFile] = File(...), authorized: bool = Depends(verify_token)):
    if authorized:
        res = dict()
        res['output'] = []
        for file in listFiles:
            file_name = file.filename
            contents = await file.read()
            nparr = np.fromstring(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            save_upload(img, file_name)
            r = detect_defect(img)
            detected_img = display_instances(img, r['rois'], r['class_ids'])
            save_detect(detected_img, file_name)
            result = {
                "image_name": file_name,
                "total_detected": len(r['scores'])
            }
            result['detected']=[]
            for i, _ in enumerate(r['scores']):
                result['detected'].append({
                    "class_name" : class_names[r['class_ids'][i]],
                    "rois" : r['rois'][i].tolist(),
                })
            res['output'].append(result)
        print(res)
        return res

@app.get("/download/image")
async def download_detected(img_name: str, authorized: bool = Depends(verify_token)):
    if authorized:
        filename = f"detected/{img_name}"
        print(img_name)
        if os.path.isfile(filename):
            return FileResponse(filename,media_type='application/octet-stream',filename=f'detected.jpg')
        else:
            return {"Error":"File not found"}

@app.get("/get-image")
async def get_image(img_name: str, authorized: bool = Depends(verify_token)):
    if authorized:
        filename = f"detected/{img_name}"
        img = cv2.imread(filename)
        h,w = img.shape[0], img.shape[1]
        img = cv2.resize(img, (1100, int(1100/w * h)))
        retval, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        return {"image":jpg_as_text}
    
    