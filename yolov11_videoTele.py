import cv2
import numpy as np
from ultralytics import YOLO
import telebot
from datetime import datetime
import time
import os
from multiprocessing import Pool, Manager

class TrafficDetector:
    def __init__(self, model_path, telegram_token, chat_id, min_vehicles=10):
        """
        Initialize the traffic detector
        model_path: Path to YOLOv8 model
        telegram_token: Telegram bot token
        chat_id: Telegram chat ID to send notifications
        min_vehicles: Minimum number of vehicles to consider as traffic jam
        """
        self.model = YOLO(model_path)  # Load the model
        self.model.to("cpu")  # Ensure it runs on CPU
        cv2.setUseOptimized(False)
        cv2.setNumThreads(1)

        self.bot = telebot.TeleBot(telegram_token)
        self.chat_id = chat_id
        self.min_vehicles = min_vehicles
        self.last_notification_time = 0
        self.notification_cooldown = 300  # 5 minutes between notifications

    def count_vehicles(self, result):
        """Count vehicles in detected objects"""
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        count = 0
        
        for box in result.boxes:
            if result.names[int(box.cls)] in vehicle_classes:
                count += 1
                
        return count

    def send_telegram_notification(self, frame, vehicle_count):
        """Send notification to Telegram with image and vehicle count"""
        current_time = time.time()
        
        if current_time - self.last_notification_time < self.notification_cooldown:
            return

        temp_image = f"traffic_jam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_image, frame)
        message = f"🚨 Traffic Jam Detected!\n📍 Vehicle Count: {vehicle_count}\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        with open(temp_image, 'rb') as photo:
            self.bot.send_photo(self.chat_id, photo, caption=message)
        
        self.last_notification_time = current_time
        os.remove(temp_image)

    def process_frame(self, frame_number, frame, frame_count):
        """Process a single frame"""
        frame = cv2.resize(frame, (640, 480))  # Resize frame to 640x480
        results = self.model(frame, stream=True)
        
        for result in results:
            vehicle_count = self.count_vehicles(result)
            if vehicle_count >= self.min_vehicles:
                self.send_telegram_notification(frame, vehicle_count)

    def process_video(self, video_path):
        """Process video file and detect traffic jams using multiprocessing"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}")
        print(f"Total frames: {frame_count}")

        frame_number = 0
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % 5 == 0:
                frames.append((frame_number, frame, frame_count))

        cap.release()

        # Use multiprocessing pool to process frames
        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap(self.process_frame, frames)

def main():
    MODEL_PATH = "best.pt"
    TELEGRAM_TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
    CHAT_ID = "@trafficitera"
    MIN_VEHICLES = 10
    VIDEO_PATH = "video.mp4"

    detector = TrafficDetector(MODEL_PATH, TELEGRAM_TOKEN, CHAT_ID, MIN_VEHICLES)
    detector.process_video(VIDEO_PATH)

if __name__ == "__main__":
    main()
