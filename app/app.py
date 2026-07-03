# Luca Kasian Koren, 12331094

# Import Shiny components
from shiny.express import input, render, ui
from shiny import reactive

# Import machine learning and image processing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from pathlib import Path


# Select GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define path to the saved model weights
MODEL_PATH = Path(__file__).parent.parent / "assets" / "weights" / "best_model.pth"

# Mapping from numeric labels to class names
LABEL_DICT = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake"
}


# CNN model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Forward pass through CNN
        x = self.features(x)
        x = self.classifier(x)
        return x


# Load model and trained weights
model = SimpleCNN(num_classes=10).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
_ = model.load_state_dict(state_dict)
_ = model.eval()


# Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.3444, 0.3802, 0.4078],
        std=[0.2037, 0.1366, 0.1148]
    )
])


# Set page title
ui.page_opts(title="Satellite Image Classification", fillable=True)

# App heading and description
ui.h2("Satellite Image Classification with CNN")
ui.p("Upload a 64x64 RGB satellite image. The model predicts the terrain class.")

# File upload input
ui.input_file("image_file", "Upload satellite image", accept=[".jpg", ".jpeg", ".png"])


@reactive.calc
def prediction_result():
    """
    Loads the uploaded image, applies preprocessing,
    runs the model prediction, and returns the result.
    """

    file = input.image_file()

    # Return nothing if no image was uploaded
    if file is None:
        return None

    # Load uploaded image
    img_path = file[0]["datapath"]
    image = Image.open(img_path).convert("RGB")

    # Apply preprocessing and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Run model inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0].cpu()

    # Get predicted class and confidence
    predicted_idx = int(torch.argmax(probabilities))
    predicted_class = LABEL_DICT[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    # Create probability table for all classes
    prob_table = pd.DataFrame({
        "Class": [LABEL_DICT[i] for i in range(10)],
        "Probability": [float(probabilities[i]) for i in range(10)]
    })

    # Format and sort probabilities
    prob_table["Probability"] = prob_table["Probability"].round(4)
    prob_table = prob_table.sort_values("Probability", ascending=False)

    return {
        "image_path": img_path,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "prob_table": prob_table
    }


@render.image
def uploaded_image():
    # Display uploaded image
    result = prediction_result()

    if result is None:
        return None

    return {
        "src": result["image_path"],
        "width": "256px",
        "height": "256px"
    }


@render.ui
def prediction_text():
    # Display predicted class and confidence
    result = prediction_result()

    if result is None:
        return ui.p("No image uploaded yet.")

    return ui.div(
        ui.h3(f"Predicted class: {result['predicted_class']}"),
        ui.h4(f"Confidence: {result['confidence']:.2%}")
    )


@render.data_frame
def probability_table():
    # Display class probabilities as a table
    result = prediction_result()

    if result is None:
        return pd.DataFrame({
            "Class": [],
            "Probability": []
        })

    return result["prob_table"]