from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import torch
import torchvision
import uvicorn
import PIL
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from pytorch_grad_cam import GradCAM,GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image

app = FastAPI()

# Allow all origins, methods, and headers for simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        # Load the pre-trained ResNet-101 model
        # resnet = torchvision.models.resnet50(pretrained=True)
        resnet = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')        
        # resnet = torch.load(r'E:\Proposal\Deepfake-detection-project\backend\resnet50-0676ba61.pth')
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Add a new fully connected layer with the desired number of classes
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



MODEL_PATH = r'E:\Proposal\Deepfake-detection-project\backend\resnet50.pt'
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
        # transforms.Normalize(mean=[0.5186, 0.4302, 0.3818], std=[0.2998, 0.2743, 0.2720]), # Resnet 9
        transforms.Normalize(mean=[0.5202, 0.4318, 0.3835], std=[0.2987, 0.2736, 0.2719]), # New 93.8
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

    targets = [ClassifierOutputTarget(0)] 
    target_layers = [model.features[-2]]# instantiate the model
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers) # use GradCamPlusPlus class

    # Preprocess input image, get the input image tensor
    img = np.array(PIL.Image.open(io.BytesIO(image_data)))
    img = cv2.resize(img, (128,128))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img)

    # generate CAM
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

    cam = np.uint8(255*grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])

    # # display the original image & the associated CAM
    # images = np.hstack((np.uint8(255*img), cam_image))
    # PIL.Image.fromarray(images)
    

    return class_labels[predicted_class_index],cam.tolist()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    prediction,cam = classify_image(contents, model, class_labels)
    # prediction2,cam_image = classify_image(contents, model, class_labels)
    return {"filename": file.filename,
            "prediction": prediction,
            "cam_image": cam
            }


if __name__ == "__main__":    
    uvicorn.run(app, host="127.0.0.1", port=8000)