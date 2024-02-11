from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# Allow all origins, methods, and headers for simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = '/home/rapzy/Downloads/Deepfake-Detection-Project/backend/scripted.pt'
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
class_labels = ['Fake', 'Real']  # Replace with your own class labels

def classify_image(image_data, MODEL_PATH, class_labels):
    # Load the model
    # model = torch.load(MODEL_PATH)
    model.eval()
    # model.to('cpu')

    # Define transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5186, 0.4302, 0.3818], std=[0.2998, 0.2743, 0.2720]),
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

    return class_labels[predicted_class_index]

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = classify_image(contents, MODEL_PATH, class_labels)
    return {"prediction": prediction,
            "filename": file.filename}