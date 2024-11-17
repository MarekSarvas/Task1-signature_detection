from fastapi import FastAPI
from app.routers.prediction import prediction_router

app = FastAPI()

app.include_router(prediction_router)

@app.get("/")
async def root():
    return {"message": "Welcome to YOLOv11 signature detector."}
