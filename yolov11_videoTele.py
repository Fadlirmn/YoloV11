import cv2
import numpy as np
from ultralytics import YOLO
import telebot
from datetime import datetime
import time
import os

class TrafficDetector:
    def __init__(self, model_path, telegram_token, chat_id, min_vehicles=10):
        """
        Initialize the traffic detector
        model_path: Path to YOLOv8 model
        telegram_token: Telegram bot token
        chat_id: Telegram chat ID to send notifications
        min_vehicles: Minimum number of vehicles to consider as traffic jam
        """
        # Force CPU-only mode for YOLO model
        self.model = YOLO(model_path)  # Load the model
        self.model.to("cpu")  # Ensure it runs on CPU

        # Disable OpenCV GPU optimizations
        cv2.setUseOptimized(False)  # Disable OpenCV optimizations
        cv2.setNumThreads(1)  # Limit to 1 CPU thread (CPU-only mode)

        self.bot = telebot.TeleBot(telegram_token)
        self.chat_id = chat_id
        self.min_vehicles = min_vehicles
        self.last_notification_time = 0
        self.notification_cooldown = 300  # 5 minutes between notifications

    def count_vehicles(self, result):
        """Count vehicles in detected objects"""
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        count = 0
        
        # Iterate through the detected boxes
        for box in result.boxes:
            # Check if the class of the detection is a vehicle
            if result.names[int(box.cls)] in vehicle_classes:
                count += 1
                
        return count

    def send_telegram_notification(self, frame, vehicle_count):
        """Send notification to Telegram with image and vehicle count"""
        current_time = time.time()
        
        # Check if enough time has passed since last notification
        if current_time - self.last_notification_time < self.notification_cooldown:
            return

        # Save frame as temporary image
        temp_image = f"traffic_jam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_image, frame)

        # Send message and image to Telegram
        message = f"🚨 Traffic Jam Detected!\n📍 Vehicle Count: {vehicle_count}\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        with open(temp_image, 'rb') as photo:
            self.bot.send_photo(self.chat_id, photo, caption=message)
        
        # Update last notification time
        self.last_notification_time = current_time
        
        # Clean up temporary image
        os.remove(temp_image)

    def process_video(self, video_path):
        """Process video file and detect traffic jams without displaying frames."""
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}")
        print(f"Total frames: {frame_count}")

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            # Process every 5th frame to improve performance
            if frame_number % 5 != 0:
                continue

            # Resize frame to lower resolution to reduce load
            frame = cv2.resize(frame, (640, 480))  # Resize frame to 640x480

            # Run YOLOv8 detection
            results = self.model(frame, stream=True)
            
            for result in results:
                # Process detections
                vehicle_count = self.count_vehicles(result)
                
                # Draw detection boxes
                annotated_frame = result.plot()
                
                # Add vehicle count and frame info to frame
                cv2.putText(
                    annotated_frame,
                    f"Vehicles: {vehicle_count} | Frame: {frame_number}/{frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Check for traffic jam and send notification
                if vehicle_count >= self.min_vehicles:
                    self.send_telegram_notification(annotated_frame, vehicle_count)

                # Comment out or remove display and key press handling
                # cv2.imshow('Traffic Detection', annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        cap.release()
        cv2.destroyAllWindows()

def main():
    MODEL_PATH = "best.pt"
    TELEGRAM_TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
    CHAT_ID = "@trafficitera"
    MIN_VEHICLES = 10
    VIDEO_PATH = "video.mp4"

    # Initialize and run detector
    detector = TrafficDetector(MODEL_PATH, TELEGRAM_TOKEN, CHAT_ID, MIN_VEHICLES)
    detector.process_video(VIDEO_PATH)

if __name__ == "__main__":2
    main()