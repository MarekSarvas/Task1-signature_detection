FROM python:3.12-slim

# Set the working dir
WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY src/sign_detector/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the code
COPY src/sign_detector/app /app/app
COPY models/yolo11m_100e_400img.pt /app/models/yolo11.pt
COPY models/english_g2.pth /app/models/english_g2.pth
COPY models/craft_mlt_25k.pth /app/models/craft_mlt_25k.pth

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
