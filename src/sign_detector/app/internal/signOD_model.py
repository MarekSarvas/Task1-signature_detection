from typing import List, Dict

from ultralytics import YOLO
import numpy as np
import cv2
import easyocr


class SignatureDetector:
    """ Yolo11 for model for detection of signed/unsigned documents
    """
    def __init__(self):
        self.model = YOLO("/app/models/yolo11.pt")
        self.ocr = reader = easyocr.Reader(['en'], model_storage_directory="/app/models", download_enabled=False)
        self.bbox_height_scale = 0.01

    def predict(self, image_data: bytes) -> List[Dict]:
        """Detects bboxes and extracts text from them. 

        Args:
            image_data (bytes): Document image.

        Returns:
            List[Dict]: Dictionaries containing bbox, class and text.
        """
        # Decode the image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = self.model(image)

        # Parse the predictions
        predictions = []
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, confidence, class_id = box.tolist()

                # Make the bounding box higher in case not whole text is in it
                y2 = int(y2)+int(self.bbox_height_scale*y2)
                text = self.extract_text(image, int(x1),int(y1), int(x2), int(y2))
               
                predictions.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "text": text,
                })

        # Classify whole document based on the found or missing signatures
        labels = list(set(prediction["class_id"] for prediction in predictions))
        if len(labels) == 1:
            doc_class = "signed" if labels[0] == 1 else "not signed"
        else:
            doc_class = "partially signed"

        return predictions, doc_class

    def extract_text(self, img, x1: int, y1: int, x2: int, y2: int) -> List[str]:
        """ Extracts text from the bbox found by the model with pre-trained ocr.

        Args:
            img : image containing detected signature
            x1 (int):
            y1 (int): 
            x2 (int):
            y2 (int):

        Returns:
            List[str]: Text detected in bounding box.
        """
        # parse bbox
        signature = img[y1:y2, x1:x2].copy()
        # run OCR
        results = self.ocr.readtext(signature)
        extracted_text = []
        # select text with confidence above threshold
        for res in results:
            text, conf = res[1], res[2]
            print(text)
            if conf > 0.5:
                extracted_text.append(text)
        return extracted_text
