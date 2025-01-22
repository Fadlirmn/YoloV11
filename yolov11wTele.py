import cv2
import numpy as np
from ultralytics import YOLO
import telebot
from datetime import datetime
import time
import os
import multiprocessing as mp
from queue import Empty
from typing import List, Tuple

class MultiprocessTrafficDetector:
    def __init__(self, model_path: str, telegram_token: str, chat_id: str, min_vehicles: int = 10):
        """
        Initialize the multiprocess traffic detector
        
        Args:
            model_path: Path to YOLOv8 model
            telegram_token: Telegram bot token
            chat_id: Telegram chat ID for notifications
            min_vehicles: Minimum vehicles to trigger traffic jam alert
        """
        self.model_path = model_path
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.min_vehicles = min_vehicles
        self.notification_cooldown = 300  # 5 minutes between notifications
        self.last_notification_time = 0

    def init_worker(self):
        """Initialize YOLO model for each worker process"""
        # Force CPU-only mode
        model = YOLO(self.model_path)
        model.to("cpu")
        return model

    def count_vehicles(self, result) -> int:
        """Count vehicles in detected objects"""
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        return sum(1 for box in result.boxes if result.names[int(box.cls)] in vehicle_classes)

    def process_frame(self, args: Tuple[np.ndarray, int, int, mp.Queue]):
        """Process a single frame in a worker process"""
        frame, frame_number, total_frames, result_queue = args
        model = self.init_worker()

        # Resize frame to reduce processing load
        frame = cv2.resize(frame, (640, 480))

        # Run detection
        results = model(frame, stream=True)
        
        for result in results:
            vehicle_count = self.count_vehicles(result)
            
            # Create annotated frame
            annotated_frame = result.plot()
            
            # Add vehicle count text
            cv2.putText(
                annotated_frame,
                f"Vehicles: {vehicle_count} | Frame: {frame_number}/{total_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Put results in queue
            result_queue.put((frame_number, annotated_frame, vehicle_count))

    def send_telegram_notification(self, frame: np.ndarray, vehicle_count: int):
        """Send notification to Telegram with image and vehicle count"""
        current_time = time.time()
        
        if current_time - self.last_notification_time < self.notification_cooldown:
            return

        temp_image = f"traffic_jam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_image, frame)

        bot = telebot.TeleBot(self.telegram_token)
        message = f"🚨 Traffic Jam Detected!\n📍 Vehicle Count: {vehicle_count}\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        with open(temp_image, 'rb') as photo:
            bot.send_photo(self.chat_id, photo, caption=message)
        
        self.last_notification_time = current_time
        os.remove(temp_image)

    def process_video(self, video_path: str):
        """Process video file using multiple processes"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")

        # Initialize multiprocessing components
        num_processes = mp.cpu_count() - 1  # Leave one CPU core free
        pool = mp.Pool(processes=num_processes)
        result_queue = mp.Manager().Queue()

        # Process frames
        frame_number = 0
        batch_size = 5  # Process every 5th frame
        processing_tasks = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % batch_size != 0:
                continue

            # Submit frame for processing
            task = pool.apply_async(
                self.process_frame, 
                args=((frame, frame_number, total_frames, result_queue),)
            )
            processing_tasks.append(task)

        # Wait for all tasks to complete
        for task in processing_tasks:
            task.get()

        # Process results in order
        results = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except Empty:
                break

        # Sort results by frame number and process notifications
        for frame_num, annotated_frame, vehicle_count in sorted(results):
            if vehicle_count >= self.min_vehicles:
                self.send_telegram_notification(annotated_frame, vehicle_count)

        # Cleanup
        pool.close()
        pool.join()
        cap.release()
        cv2.destroyAllWindows()

def main():
    MODEL_PATH = "best.pt"
    TELEGRAM_TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
    CHAT_ID = "-2294722259"
    MIN_VEHICLES = 10
    VIDEO_PATH = "video.mp4"

    detector = MultiprocessTrafficDetector(MODEL_PATH, TELEGRAM_TOKEN, CHAT_ID, MIN_VEHICLES)
    detector.process_video(VIDEO_PATH)

if __name__ == "__main__":
    main()