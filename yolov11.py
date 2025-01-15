from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("best.pt")

# Define path to video file
source = "path/to/video.mp4"

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects