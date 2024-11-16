# Dataset
1. Download data from kaggle into **data/** directory:
    - signatures https://www.kaggle.com/datasets/robinreni/signature-verification-dataset
    - documents https://www.kaggle.com/datasets/victordibia/signverod/data
2. Create signatures that will be inserted into the documents: 
```bash
python src/data/clear_signatures.py
python src/data/create_signature_class_imgs.py
```
3. Create final dataset for the yolo object detection training:
```bash
python src/data/create_dataset.py
```

# Object detection
1. Create model directory for trained models.
```bash
mkdir models
```
Train yolo for detectin signed and unsigned documents:
```bash
python src/train_od.py
```


# Results
Yolo11 medium model trained on 400 images for 100 epochs.
Model | P     | R     | mAP50 | mAP50-95 |
|-------|-------|-------|-------|----------|
yolo11m_100e_400img | 0.979 | 0.973 | 0.991 | 0.812    |