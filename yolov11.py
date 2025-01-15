import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path

def predict_video(model_path, video_path, output_path=None, conf_threshold=0.25):
    """
    Run YOLOv8 inference on a video file
    
    Args:
        model_path (str): Path to the YOLOv8 model weights
        video_path (str): Path to input video file
        output_path (str, optional): Path to save output video
        conf_threshold (float): Confidence threshold for detections (0-1)
    """
    # Load the YOLOv8 model
    model = YOLO(best.pt)
    
    # Open the video file
    cap = cv2.VideoCapture(video.mp4)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer if specified
    if output_path:
        output_path = Path(output_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference on the frame
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                show=False
            )
            
            # Draw the results on the frame
            annotated_frame = results[0].plot()
            
            # Display the frame
            cv2.imshow('YOLOv8 Detection', annotated_frame)
            
            # Save the frame if output path is specified
            if output_path:
                out.write(annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Paths configuration
    model_path = "best.pt"  # Path to your trained model or use pretrained like 'yolov8n.pt'
    video_path = "video.mp4"        # Path to your input video
    output_path = "output_video.mp4"             # Path for saving the output video
    
    # Run detection
    predict_video(
        model_path=model_path,
        video_path=video_path,
        output_path=output_path,
        conf_threshold=0.25  # Adjust confidence threshold as needed
    )