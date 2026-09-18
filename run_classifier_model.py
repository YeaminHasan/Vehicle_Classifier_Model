import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

from pathlib import Path
import cv2
from ultralytics import YOLO

os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

MODEL_PATH = "weights/best.pt"
TEST_IMAGES_DIR = Path("test_images")
IMAGE_SIZE = (800, 640)
VEHICLE_CLASSES = {"car", "threewheel", "bus", "truck", "motorbike", "van"}


def predict_and_display(model, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return True

    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    results = model.predict(source=image, conf=0.4, verbose=False)[0]

    print(f"\n--- Detection Results: {image_path.name} ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}) ---")
    detection_count = 0
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = str(model.names[class_id])

        if class_name.lower() not in VEHICLE_CLASSES:
            continue

        detection_count += 1
        label = f"{class_name}: {confidence:.2f}"
        print(f"Detected: {class_name} | Confidence: {confidence:.2f}")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_origin = (x1, max(y1 - 10, 20))
        cv2.putText(
            image,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if detection_count == 0:
        print("No Car, Threewheel, Bus, Truck, Motorbike, or Van detected.")

    cv2.imshow("Vehicle Detection - VS Code Test", image)
    return cv2.waitKey(0) & 0xFF != ord("q")


if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    image_paths = sorted(
        path
        for path in TEST_IMAGES_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {TEST_IMAGES_DIR}")

    try:
        for image_path in image_paths:
            if not predict_and_display(model, image_path):
                break
    finally:
        cv2.destroyAllWindows()