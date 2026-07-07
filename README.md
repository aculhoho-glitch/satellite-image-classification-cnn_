# Satellite Image Classification with CNN

This project uses a deep learning approach, specifically a Convolutional Neural Network (CNN), to classify satellite images into 10 land-use categories.

The project includes a Jupyter Notebook for training and evaluating the model, a trained PyTorch model, and a Shiny app that allows users to upload satellite images and receive a predicted class.

## Classes

The model predicts one of the following classes:

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

## Project Structure

```text
satellite-image-classification-cnn/
│
├── app/
│   └── app.py
│
├── assets/
│   └── weights/
│       └── best_model.pth
│
├── notebooks/
│   └── satellite-image-classification-cnn.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Files

### `notebooks/satellite-image-classification-cnn.ipynb`

This notebook contains the complete model development process, including:

* loading the satellite image dataset
* preprocessing the images
* splitting the data into training and validation sets
* defining the CNN architecture
* training the model
* evaluating the model
* saving the best model weights

### `app/app.py`

This file contains the Shiny app.

The app allows the user to upload a satellite image. The image is resized to 64 × 64 pixels, processed in the same way as during training, and then classified by the trained CNN model.

### `assets/weights/best_model.pth`

This file contains the saved weights of the trained CNN model.

The Shiny app loads this file to make predictions.

### `requirements.txt`

This file contains the Python packages needed to run the project.

## Installation

First, download or clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/satellite-image-classification-cnn.git
```

Then open the project folder in CMD:

```bash
cd satellite-image-classification-cnn
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Open the Notebook

Open CMD inside the project folder and start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/satellite-image-classification-cnn.ipynb
```

## How to Run the Shiny App

Open CMD inside the main project folder:

```bash
cd satellite-image-classification-cnn
```

Then run the app with:

```bash
python -m shiny run --reload app/app.py
```

After starting the app, CMD will show a local link, usually similar to:

```text
http://127.0.0.1:8000
```

Open this link in your browser.

## How the App Works

1. The user uploads a satellite image.
2. The image is resized to 64 × 64 pixels.
3. The image is converted into a tensor and normalized.
4. The trained CNN model loads the saved weights from `assets/weights/best_model.pth`.
5. The model predicts one of the 10 satellite image classes.
6. The predicted class is displayed in the Shiny app.

## Model

The model is a Convolutional Neural Network implemented with PyTorch.

It was trained to classify satellite images into 10 land-use categories. The best model weights are saved in:

```text
assets/weights/best_model.pth
```

## Notes

The dataset itself is not included in this repository because image datasets can be large.

The repository focuses on:

* the training notebook
* the trained model
* the Shiny app
* the required project files

## Author

Luca Kasian Koren
