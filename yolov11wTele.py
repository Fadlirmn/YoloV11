import cv2
import numpy as np
from ultralytics import YOLO
import telebot
from datetime import datetime
import time
import os
import multiprocessing as mp
from queue import Empty
from typing import List, Tuple, Dict
from threading import Thread

class TelegramBot:
    def __init__(self, token: str, min_vehicles: int = 10):
        self.bot = telebot.TeleBot(token)
        self.active_groups = set()
        self.min_vehicles = min_vehicles
        self.setup_handlers()

    def setup_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def start(message):
            if message.chat.type in ['group', 'supergroup']:
                self.active_groups.add(message.chat.id)
                welcome_message = (
                    "🚦 Traffic Monitoring Bot activated!\n\n"
                    "Commands:\n"
                    "/start - Start monitoring\n"
                    "/stop - Stop monitoring\n"
                    "/status - Check monitoring status\n"
                    "/set_min_vehicles [number] - Set minimum vehicles threshold"
                )
                self.bot.reply_to(message, welcome_message)

        @self.bot.message_handler(commands=['stop'])
        def stop(message):
            if message.chat.id in self.active_groups:
                self.active_groups.remove(message.chat.id)
                self.bot.reply_to(message, "❌ Traffic monitoring stopped for this group.")

        @self.bot.message_handler(commands=['status'])
        def status(message):
            status_message = "✅ Monitoring active" if message.chat.id in self.active_groups else "❌ Monitoring inactive"
            self.bot.reply_to(message, status_message)

    def send_notification(self, image_path: str, vehicle_count: int):
        message = (
            f"🚨 Traffic Jam Detected!\n"
            f"📍 Vehicle Count: {vehicle_count}\n"
            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        for group_id in self.active_groups:
            try:
                with open(image_path, 'rb') as photo:
                    self.bot.send_photo(group_id, photo, caption=message)
            except Exception as e:
                print(f"Error sending notification to group {group_id}: {e}")

    def start_polling(self):
        self.bot.polling(none_stop=True)

def process_frame(frame_data: Tuple[np.ndarray, int, int, str, YOLO]) -> Tuple[int, str, int]:
    """Process a single frame"""
    frame, frame_number, total_frames, model_path = frame_data
    model = YOLO(model_path)
    model.to("cpu")

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)
    
    vehicle_count = 0
    annotated_frame = None
    
    for result in results:
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        vehicle_count = sum(1 for box in result.boxes if result.names[int(box.cls)] in vehicle_classes)
        annotated_frame = result.plot()
        
        cv2.putText(
            annotated_frame,
            f"Vehicles: {vehicle_count} | Frame: {frame_number}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Save frame to temporary file
    temp_image = f"frame_{frame_number}.jpg"
    cv2.imwrite(temp_image, annotated_frame)
    
    return frame_number, temp_image, vehicle_count

class TrafficDetector:
    def __init__(self, model_path: str, telegram_token: str, min_vehicles: int = 10):
        self.model_path = model_path
        self.min_vehicles = min_vehicles
        self.notification_cooldown = 300
        self.last_notification_time = 0
        
        # Initialize Telegram bot in a separate thread
        self.telegram_bot = TelegramBot(telegram_token, min_vehicles)
        self.bot_thread = Thread(target=self.telegram_bot.start_polling)
        self.bot_thread.daemon = True
        self.bot_thread.start()

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")

        # Initialize multiprocessing pool
        num_processes = mp.cpu_count() - 1
        pool = mp.Pool(processes=num_processes)
        
        frame_number = 0
        batch_size = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % batch_size != 0:
                continue

            # Process frame asynchronously
            frame_data = (frame, frame_number, total_frames, self.model_path)
            result = pool.apply_async(process_frame, (frame_data,))
            
            try:
                frame_num, temp_image_path, vehicle_count = result.get()
                
                # Check for traffic jam and send notification
                current_time = time.time()
                if (vehicle_count >= self.min_vehicles and 
                    current_time - self.last_notification_time >= self.notification_cooldown):
                    self.telegram_bot.send_notification(temp_image_path, vehicle_count)
                    self.last_notification_time = current_time
                
                # Clean up temporary image
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    
            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}")

        pool.close()
        pool.join()
        cap.release()
        cv2.destroyAllWindows()

def main():
    MODEL_PATH = "best.pt"
    TELEGRAM_TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
    MIN_VEHICLES = 10
    VIDEO_PATH = "video.mp4"

    detector = TrafficDetector(MODEL_PATH, TELEGRAM_TOKEN, MIN_VEHICLES)
    detector.process_video(VIDEO_PATH)

if __name__ == "__main__":
    main()