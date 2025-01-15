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
        self.model = YOLO(model_path)
        self.bot = telebot.TeleBot(telegram_token)
        self.chat_id = chat_id
        self.min_vehicles = min_vehicles
        self.last_notification_time = 0
        self.notification_cooldown = 300  # 5 minutes between notifications

    def count_vehicles(self, detections):
        """Count vehicles in detected objects"""
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        count = 0
        for detection in detections:
            if detection.cls in vehicle_classes:
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

    def process_camera_feed(self):
        """Process live camera feed and detect traffic jams"""
        # Initialize camera
        cap = cv2.VideoCapture('video.mp4')  # Use 0 for default camera, or specify IP camera URL
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 detection
            results = self.model(frame, stream=True)
            
            for result in results:
                # Process detections
                vehicle_count = self.count_vehicles(result)
                
                # Draw detection boxes
                annotated_frame = result.plot()
                
                # Add vehicle count to frame
                cv2.putText(
                    annotated_frame,
                    f"Vehicles: {vehicle_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Check for traffic jam and send notification
                if vehicle_count >= self.min_vehicles:
                    self.send_telegram_notification(annotated_frame, vehicle_count)

                # Display the frame
                cv2.imshow('Traffic Detection', annotated_frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    # Configuration
    MODEL_PATH = "best.pt"  # or path to your custom trained model
    TELEGRAM_TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
    CHAT_ID = "@trafficitera"
    MIN_VEHICLES = 10  # Adjust this threshold based on your needs

    # Initialize and run detector
    detector = TrafficDetector(MODEL_PATH, TELEGRAM_TOKEN, CHAT_ID, MIN_VEHICLES)
    detector.process_camera_feed()

if __name__ == "__main__":
    main()