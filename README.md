# Vehicle Classifier Model

A vehicle detection project built with [Ultralytics YOLO](https://docs.ultralytics.com/) and OpenCV. The project loads a trained YOLO model, detects vehicles in test images, draws bounding boxes, and displays the results.

## Features

- Detects vehicles using a custom-trained YOLO model
- Supports batch testing of images from `test_images/`
- Displays detected class names and confidence scores
- Draws bounding boxes around detected vehicles
- Supports the following vehicle classes:
  - Car
  - Threewheel
  - Bus
  - Truck
  - Motorbike
  - Van

## Project Structure

```text
Vehicle_Classifier_Model/
├── requirements.txt
├── run_classifier_model.py
├── test_images/
│   └── *.png, *.jpg, *.jpeg, *.bmp
└── weights/
    └── best.pt
```

## Requirements

- Python 3.9 or newer
- A desktop environment with GUI support for OpenCV image display
- The trained model file at `weights/best.pt`

## Installation

1. Clone or download this repository and open its directory:

   ```bash
   cd Vehicle_Classifier_Model
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Add supported image files to the `test_images/` directory.
2. Make sure the trained model is available at `weights/best.pt`.
3. Run the test script:

   ```bash
   python run_classifier_model.py
   ```

The script processes the images one by one and opens a result window for each image. Press any key to continue to the next image. Press `q` to stop testing.

## Detection Settings

- Image size: `800 x 640`
- Confidence threshold: `0.40`
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

These settings can be changed in `run_classifier_model.py`.

## Troubleshooting

### No images found

Make sure at least one supported image exists inside the `test_images/` directory.

### Model file not found

Verify that the trained model exists at:

```text
weights/best.pt
```

### OpenCV window does not appear

Run the script in a desktop session with GUI support. `opencv-python` is required because the script uses `cv2.imshow()`.
