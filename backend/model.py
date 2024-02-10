import torch
from torchvision import transforms
from PIL import Image

model_path = '/home/rapzy/Downloads/LearnPython/FastApi/scripted.pt'
image_path = '/home/rapzy/Downloads/monu.png'
class_labels = ['fake', 'real']  # Replace with your own class labels

def classify_image(image_path, model_path, class_labels):
    # Load the model
    # model = torch.load(model_path)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    # model.to('cpu')

    # Define transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5186, 0.4302, 0.3818], std=[0.2998, 0.2743, 0.2720]),
    ])

    # Load and preprocess the image
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        output = model(img)

    # Get predicted class index
    _, predicted = torch.max(output, 1)
    predicted_class_index = predicted.item()

    return class_labels[predicted_class_index]

# Example usage:
predicted_class = classify_image(image_path, model_path, class_labels)
print("Predicted Class:", predicted_class)
