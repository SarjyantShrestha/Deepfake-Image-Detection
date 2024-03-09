from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
import PIL
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io


app = FastAPI()

# Cross-Origin Resource Sharing (CORS)
""" CORS is a security feature implemented by web browsers to restrict web pages
from making requests to a different domain than the one that served the original
web page. CORS middleware allows you to relax these restrictions and define
which origins, methods, and headers are allowed for requests to your
application."""
app.add_middleware(
    CORSMiddleware,
    # Requests from any origin will be allowed
    allow_origins=["*"],
    # Determines whether the browser should include credentials (such as cookies or HTTP authentication) with cross-origin requests.
    allow_credentials=True,
    # HTTP methods are allowed for cross-origin requests.
    allow_methods=["*"],
    # All HTTP headers are allowed in cross-origin request.
    allow_headers=["*"],
)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 3*128*128
        self.conv1 = self.conv_block(in_channels, 64)
        # 64*128*128
        self.conv2 = self.conv_block(64, 128, pool=True)
        # 128*64*64
        self.res1 = nn.Sequential(self.conv_block(
            128, 128), self.conv_block(128, 128))
        # 128*64*64
        self.conv3 = self.conv_block(128, 256, pool=True)
        # 256*32*32
        self.conv4 = self.conv_block(256, 512, pool=True)
        # 512*16*16
        self.res2 = nn.Sequential(self.conv_block(
            512, 512), self.conv_block(512, 512))
        # 512*16*16
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        # 512*4*4
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512*4*4, num_classes))

    def conv_block(self, in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


MODEL_PATH = "/home/rapzy/Downloads/Deepfake-Image-Detection/backend/resnet9_scripted.pt"
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
class_labels = ['Fake', 'Real']  # Replace with your own class labels


def classify_image(image_data, model, class_labels):
    # Load the model
    # model = torch.load(MODEL_PATH)
    model.eval()
    # model.to('cpu')

    # Define transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5186, 0.4302, 0.3818], std=[
        #                      0.2998, 0.2743, 0.2720]),  # Resnet 9
        transforms.Normalize(mean=[0.5202, 0.4318, 0.3835], std=[
                             0.2987, 0.2736, 0.2719]),  # New 93.8
        # transforms.Normalize(mean=[0.5274, 0.4384, 0.3886], std=[
        #  0.2999, 0.2781, 0.2768]),  # New 93.8
    ])

    # Load and preprocess the image
    img = Image.open(io.BytesIO(image_data))
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        output = model(img)

    # Get predicted class index
    _, predicted = torch.max(output, 1)
    predicted_class_index = predicted.item()

    # Preprocess input image, get the input image tensor
    img = np.array(PIL.Image.open(io.BytesIO(image_data)))
    img = cv2.resize(img, (128, 128))
    img = np.float32(img) / 255

    return class_labels[predicted_class_index]


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = classify_image(contents, model, class_labels)
    return {"filename": file.filename,
            "prediction": prediction,
            }
