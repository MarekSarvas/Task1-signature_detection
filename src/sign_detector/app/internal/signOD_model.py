from ultralytics import YOLO
import numpy as np
import cv2


class SignatureDetector:
    """ Yolo11 for model for detection of signed/unsigned documents
    """
    def __init__(self):
        self.model = YOLO("/app/models/yolo11.pt")

    def predict(self, image_data: bytes):
        # Decode the image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = self.model(image)

        # Parse the predictions
        predictions = []
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, confidence, class_id = box.tolist()
                predictions.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": float(confidence),
                    "class_id": int(class_id)
                })

        # Classify whole document based on the found or missing signatures
        labels = list(set(prediction["class_id"] for prediction in predictions))
        if len(labels) == 1:
            doc_class = "signed" if labels[0] == 1 else "not signed"
        else:
            doc_class = "partialy signed"

        return predictions, doc_class
