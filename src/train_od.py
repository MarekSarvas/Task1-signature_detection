from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.pt")

# Train the model
train_results = model.train(
    data="data/FINAL_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    device="cuda",
    batch=8,
    patience=5,
)

# Evaluate model performance on the validation set
model.save("models/yolo11m_100e_400img.pt")

metrics = model.val(data="data/FINAL_dataset/data_test.yaml")
