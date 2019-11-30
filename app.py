import io
import os
import json
import numpy as np
import cv2
import base64
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, url_for




app = Flask(__name__)

# url_for('static', filename='liz.pt')

# model = torch.load("/home/eric/Dropbox/projects/programming/2019/Python_progs/pytorch/oven_project/liz.pt");
model = torch.load('static/liz.pt')
model.eval()
class_names = ['off', 'on', 'undefined']

## read in validation file names

# PATH_ON = "/home/eric/Dropbox/projects/programming/2019/Python_progs/pytorch/oven_project/ovenflask/static/liz/val/on"
# PATH_OFF = "/home/eric/Dropbox/projects/programming/2019/Python_progs/pytorch/oven_project/ovenflask/static/liz/val/off"


# app.config['OVENOFF_FOLDER'] = os.path.join('static', 'liz', 'val', 'off')
# app.config['OVENON_FOLDER'] = os.path.join('static', 'liz', 'val', 'on')

# oven_on_filenames = [os.path.join(PATH_ON,v) for v in os.listdir(PATH_ON) ]
# oven_off_filenames = [os.path.join(PATH_OFF,v) for v in os.listdir(PATH_OFF) ]

# oven_onoff_filenames = oven_on_filenames+oven_off_filenames


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
  
    print("********* torch ouputs ***********")
    # print(outputs[0], outputs[[0]])
    vals, preds = torch.max(outputs, 1)
    print(preds, vals, outputs.detach().numpy())
    aaa = outputs.detach().numpy()

    print(aaa.shape,aaa)
    print("preds[0]", preds[0])
    print("class_names[preds[0]]",class_names[preds[0]])
    
    prediction = class_names[preds[0]]
    print("prediction", prediction)

    return({'prediction':prediction})


# def predict_from_camera(img_jpg_buffer):


#     prediction = get_prediction(img_jpg_buffer)
#     return {'predicted': prediction}



@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')



@app.route('/takepic',methods=["POST"])
def disp_pic():
  
    hdr, encoded_data = request.json['image'].split(',')
    nparr = np.fromstring(base64.standard_b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # encode
    is_success, buffer = cv2.imencode(".jpg", img)

    return_vals = get_prediction(buffer)

    print("return_vals", return_vals)

    return jsonify(return_vals)

if __name__ == '__main__':
    app.run()