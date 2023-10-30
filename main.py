from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import os
from PIL import Image

import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

# インスタンス化
app = FastAPI()

#前処理
transform=transforms.Compose([transforms.ToTensor()])

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

from torchvision.models.resnet import resnet152
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet152(pretrained=True)
        self.fc = nn.Linear(1000, 3)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

net = Net().cpu().eval()

# Google Driveから学習済みモデルをダウンロードするエンドポイント
@app.get("/download_model")
async def download_model():
    # Google Driveのファイル共有リンクを直接指定してダウンロード
    url = "https://drive.google.com/uc?export=download&id=1U5aFnZtSxT8bDLpxYmekOq_aenM8vMAz"
    response = requests.get(url)

    # ファイルを保存
    with open("Assari_classification_remBG_Augummentation.pt", "wb") as f:
        f.write(response.content)

    # PyTorchでモデルを読み込む
    #model = torch.load("Asari_Classification.pt")
    net.load_state_dict(torch.load('Assari_classification_remBG_Augummentation.pt', map_location=torch.device('cpu')))
    return {"message": "モデルをダウンロードして読み込みました。"}


@app.post('/predict')
async def make_predictions(file: UploadFile = File(...)):
    contents=await file.read()
    image = Image.open(io.BytesIO(contents))
    img=transform(image)
    t=net(img.unsqueeze(0))
    y = F.softmax(t)
    y_pred=torch.argmax(y).item()
    y_pred_prob = y.squeeze().tolist()
    return {"prediction": y_pred, "probability": y_pred_prob}

 
    
    


    