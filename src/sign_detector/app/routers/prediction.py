from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from app.internal.signOD_model import SignatureDetector


prediction_router = APIRouter()

detector_model = SignatureDetector()


@prediction_router.post("/predict")
async def predict(file: UploadFile):
    try:
        # Read the image and run predictions
        image = await file.read()
        predictions, doc_class = detector_model.predict(image)


        return JSONResponse(content={"predictions": predictions,
                                     "document_class": doc_class},
                            status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
