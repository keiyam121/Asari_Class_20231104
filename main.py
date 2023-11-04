from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# インスタンス化
app = FastAPI()

# 学習済みモデルの読み込み
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = models.resnet152(pretrained=True)
        self.fc = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x

model = Net()
model.load_state_dict(torch.load('Assari_classification_rembg.pt', map_location=torch.device('cpu')))
model.eval()

# 前処理の定義
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post('/predict/')
async def make_predictions(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img)

        probs = F.softmax(output[0], dim=0)
        pred_class = torch.argmax(probs).item()

        return JSONResponse(content={"prediction": pred_class, "probability": probs[pred_class].item()})
    
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

    
