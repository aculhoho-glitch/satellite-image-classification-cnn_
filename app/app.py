from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from shiny import App, ui, render, reactive


# --------------------------------------------------

# Choose GPU if it is available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Go from the app folder back to the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the saved model weights
MODEL_PATH = BASE_DIR / "assets" / "weights" / "best_model.pth"

# Class names in the same order as the training labels
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


# CNN model for the image classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional part of the model
        # This part extracts useful image features
        self.features = nn.Sequential(
            # First block: input image has 3 channels because it is RGB
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # image size: 64 -> 32

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # image size: 32 -> 16

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # image size: 16 -> 8

            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)    # image size: 8 -> 4
        )

        # Classification part of the model
        # This part converts the extracted image features into class predictions
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 256 feature maps with size 4 x 4 are passed to the linear layer
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),

            # Dropout was used during training to reduce overfitting
            nn.Dropout(0.4),

            # Final output layer: one value for each class
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # First extract image features
        x = self.features(x)

        # Then classify the image
        x = self.classifier(x)

        return x


# Create model object
model = SimpleCNN(num_classes=10).to(DEVICE)

# Load the trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# Set model to evaluation mode for prediction
model.eval()


# Transform used for uploaded images before prediction
image_transform = transforms.Compose([
    # Every uploaded image is resized to 64 x 64 pixels
    transforms.Resize((64, 64)),

    # Convert image to tensor so PyTorch can use it
    transforms.ToTensor(),

    # Use the same normalization values as in training
    transforms.Normalize(
        mean=[0.3444, 0.3802, 0.4078],
        std=[0.2037, 0.1366, 0.1148]
    )
])


# User interface of the Shiny app
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(
            """
            body {
                background: #f4f7fb;
                font-family: Arial, sans-serif;
            }

            .hero {
                background: linear-gradient(135deg, #111827, #2563eb);
                color: white;
                padding: 35px;
                border-radius: 20px;
                margin-top: 25px;
                margin-bottom: 25px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }

            .hero h1 {
                font-size: 38px;
                font-weight: 800;
                margin-bottom: 8px;
            }

            .hero p {
                font-size: 17px;
                opacity: 0.95;
                margin-bottom: 0;
            }

            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                margin-bottom: 24px;
            }

            .card {
                background: white;
                border-radius: 18px;
                padding: 24px;
                box-shadow: 0 8px 22px rgba(0,0,0,0.08);
                border: 1px solid #e5e7eb;
            }

            .card h3 {
                margin-top: 0;
                margin-bottom: 16px;
                font-size: 22px;
                font-weight: 700;
                color: #111827;
            }

            .small-text {
                color: #6b7280;
                font-size: 14px;
                margin-top: 12px;
            }

            .placeholder {
                color: #6b7280;
                text-align: center;
                padding: 36px;
                border: 2px dashed #d1d5db;
                border-radius: 16px;
                background: #f9fafb;
            }

            .prediction-box {
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 18px;
                padding: 28px;
                text-align: center;
            }

            .prediction-label {
                color: #6b7280;
                font-size: 15px;
                margin-bottom: 8px;
            }

            .prediction-class {
                color: #1d4ed8;
                font-size: 38px;
                font-weight: 800;
                margin-bottom: 10px;
            }

            .confidence {
                color: #111827;
                font-size: 22px;
                font-weight: 700;
            }

            .prob-item {
                margin-bottom: 16px;
            }

            .prob-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 6px;
                font-size: 14px;
            }

            .prob-label {
                font-weight: 600;
                color: #111827;
            }

            .prob-value {
                color: #374151;
            }

            .prob-track {
                height: 13px;
                background: #e5e7eb;
                border-radius: 999px;
                overflow: hidden;
            }

            .prob-fill {
                height: 100%;
                background: linear-gradient(90deg, #2563eb, #60a5fa);
                border-radius: 999px;
            }

            img {
                width: 100%;
                max-height: 380px;
                object-fit: contain;
                border-radius: 16px;
                background: #f9fafb;
                border: 1px solid #e5e7eb;
            }

            .footer {
                text-align: center;
                color: #6b7280;
                font-size: 13px;
                margin-top: 20px;
                margin-bottom: 25px;
            }

            @media (max-width: 900px) {
                .grid {
                    grid-template-columns: 1fr;
                }
            }
            """
        )
    ),

    # Main header of the app
    ui.div(
        ui.h1("Satellite Image Classification"),
        ui.p("Upload a satellite image and let the trained CNN model predict the land-use class."),
        class_="hero"
    ),

    # First row with upload field and image preview
    ui.div(
        ui.div(
            ui.h3("Upload Image"),

            # File input for one image
            ui.input_file(
                "uploaded_image",
                "Choose a satellite image",
                accept=[".jpg", ".jpeg", ".png"],
                multiple=False
            ),

            ui.p(
                "Images are automatically resized to 64 × 64 pixels before prediction.",
                class_="small-text"
            ),
            class_="card"
        ),

        ui.div(
            ui.h3("Image Preview"),

            # Shows the uploaded image
            ui.output_image("preview_image"),
            class_="card"
        ),

        class_="grid"
    ),

    # Second row with prediction and probabilities
    ui.div(
        ui.div(
            ui.h3("Prediction Result"),

            # Shows predicted class and confidence
            ui.output_ui("prediction_output"),
            class_="card"
        ),

        ui.div(
            ui.h3("Top 5 Probabilities"),

            # Shows the five highest class probabilities
            ui.output_ui("probability_output"),
            class_="card"
        ),

        class_="grid"
    ),

    # Small footer at the bottom of the page
    ui.div(
        "Model: SimpleCNN | Input size: 64 × 64 | Classes: 10",
        class_="footer"
    )
)


def server(input, output, session):

    @reactive.calc
    def make_prediction():
        # Get uploaded image information
        file_info = input.uploaded_image()

        # No prediction is made before an image is uploaded
        if not file_info:
            return None

        # Temporary path of the uploaded image
        image_path = file_info[0]["datapath"]

        # Open image and make sure it has RGB channels
        image = Image.open(image_path).convert("RGB")

        # Store original image size for showing it in the result
        original_size = image.size

        # Apply resizing, tensor conversion and normalization
        image_tensor = image_transform(image)

        # Add batch dimension because the model expects a batch of images
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # Run model prediction
        with torch.no_grad():
            outputs = model(image_tensor)

            # Convert model outputs into probabilities
            probabilities = F.softmax(outputs, dim=1)[0]

        # Move probabilities back to CPU
        probabilities = probabilities.cpu()

        # Get the class with the highest probability
        predicted_index = torch.argmax(probabilities).item()
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = probabilities[predicted_index].item()

        # Save all class probabilities in a list
        all_probs = []

        for i, prob in enumerate(probabilities):
            all_probs.append((CLASS_NAMES[i], prob.item()))

        # Sort classes by probability, highest first
        all_probs = sorted(all_probs, key=lambda x: x[1], reverse=True)

        # Return all values needed in the output sections
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": all_probs,
            "original_size": original_size
        }

    @output
    @render.image
    def preview_image():
        # Get uploaded image
        file_info = input.uploaded_image()

        # If there is no image, the preview stays empty
        if not file_info:
            return None

        image_path = file_info[0]["datapath"]

        # Return image path so Shiny can display it
        return {
            "src": image_path,
            "width": "100%",
            "alt": "Uploaded satellite image"
        }

    @output
    @render.ui
    def prediction_output():
        # Get prediction result
        result = make_prediction()

        # Show this text before the first upload
        if result is None:
            return ui.div(
                "Upload an image to see the prediction.",
                class_="placeholder"
            )

        # Convert confidence to percent
        confidence_percent = result["confidence"] * 100

        # Show predicted class, confidence and original image size
        return ui.div(
            ui.div("Predicted class", class_="prediction-label"),
            ui.div(result["predicted_class"], class_="prediction-class"),
            ui.div(f"Confidence: {confidence_percent:.2f}%", class_="confidence"),
            ui.p(
                f"Original image size: {result['original_size'][0]} × {result['original_size'][1]} pixels",
                class_="small-text"
            ),
            class_="prediction-box"
        )

    @output
    @render.ui
    def probability_output():
        # Get prediction result
        result = make_prediction()

        # Show placeholder before upload
        if result is None:
            return ui.div(
                "Class probabilities will appear here after uploading an image.",
                class_="placeholder"
            )

        bars = []

        # Create a progress bar for the top five classes
        for class_name, probability in result["probabilities"][:5]:
            percent = probability * 100

            bars.append(
                ui.div(
                    ui.div(
                        ui.span(class_name, class_="prob-label"),
                        ui.span(f"{percent:.2f}%", class_="prob-value"),
                        class_="prob-row"
                    ),
                    ui.div(
                        ui.div(
                            class_="prob-fill",
                            style=f"width: {percent}%;"
                        ),
                        class_="prob-track"
                    ),
                    class_="prob-item"
                )
            )

        return ui.div(*bars)


# Create the Shiny app
app = App(app_ui, server)