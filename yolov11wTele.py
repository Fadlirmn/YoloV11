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
        self.last_vehicle_count = 0
        self.last_detection_time = None
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
                    "/current - View current traffic status\n"
                    "/set_min_vehicles [number] - Set minimum vehicles threshold\n"
                    "/help - Show all commands"
                )
                self.bot.reply_to(message, welcome_message)

        @self.bot.message_handler(commands=['stop'])
        def stop(message):
            if message.chat.id in self.active_groups:
                self.active_groups.remove(message.chat.id)
                self.bot.reply_to(message, "❌ Traffic monitoring stopped for this group.")

        @self.bot.message_handler(commands=['status'])
        def status(message):
            status_message = (
                "✅ Monitoring active\n"
                f"🚗 Current threshold: {self.min_vehicles} vehicles"
            ) if message.chat.id in self.active_groups else "❌ Monitoring inactive"
            self.bot.reply_to(message, status_message)

        @self.bot.message_handler(commands=['current'])
        def current(message):
            if message.chat.id not in self.active_groups:
                self.bot.reply_to(message, "❌ Monitoring is not active in this group.")
                return

            traffic_status = "🔴 Heavy" if self.last_vehicle_count >= self.min_vehicles else "🟢 Normal"
            detection_time = self.last_detection_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_detection_time else "No detection yet"
            
            status_message = (
                "🔄 Current Traffic Status\n"
                f"🚦 Status: {traffic_status}\n"
                f"🚗 Vehicles detected: {self.last_vehicle_count}\n"
                f"⚠️ Alert threshold: {self.min_vehicles}\n"
                f"⏰ Last detection: {detection_time}"
            )
            self.bot.reply_to(message, status_message)

        @self.bot.message_handler(commands=['set_min_vehicles'])
        def set_min_vehicles(message):
            try:
                new_threshold = int(message.text.split()[1])
                if new_threshold < 1:
                    raise ValueError
                self.min_vehicles = new_threshold
                self.bot.reply_to(message, f"✅ Alert threshold updated to {new_threshold} vehicles")
            except (IndexError, ValueError):
                self.bot.reply_to(message, "❌ Please provide a valid number (e.g., /set_min_vehicles 10)")

        @self.bot.message_handler(commands=['help'])
        def help(message):
            help_message = (
                "📋 Available Commands:\n\n"
                "🟢 /start - Begin traffic monitoring\n"
                "🔴 /stop - Stop monitoring\n"
                "ℹ️ /status - Check monitoring status\n"
                "🔄 /current - View real-time traffic status\n"
                "⚙️ /set_min_vehicles [number] - Set vehicle threshold\n"
                "❓ /help - Show this message"
            )
            self.bot.reply_to(message, help_message)

    def update_status(self, vehicle_count: int):
        self.last_vehicle_count = vehicle_count
        self.last_detection_time = datetime.now()

    def send_notification(self, image_path: str, vehicle_count: int):
        self.update_status(vehicle_count)
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

def process_frame(frame_data: Tuple[np.ndarray, int, int, str]) -> Tuple[int, str, int]:
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

    temp_image = f"frame_{frame_number}.jpg"
    cv2.imwrite(temp_image, annotated_frame)
    
    return frame_number, temp_image, vehicle_count

class TrafficDetector:
    def __init__(self, model_path: str, telegram_token: str, min_vehicles: int = 10):
        self.model_path = model_path
        self.min_vehicles = min_vehicles
        self.notification_cooldown = 300
        self.last_notification_time = 0
        
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

            frame_data = (frame, frame_number, total_frames, self.model_path)
            result = pool.apply_async(process_frame, (frame_data,))
            
            try:
                frame_num, temp_image_path, vehicle_count = result.get()
                self.telegram_bot.update_status(vehicle_count)
                
                current_time = time.time()
                if (vehicle_count >= self.min_vehicles and 
                    current_time - self.last_notification_time >= self.notification_cooldown):
                    self.telegram_bot.send_notification(temp_image_path, vehicle_count)
                    self.last_notification_time = current_time
                
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    
            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}")

        pool.close()
        pool.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = "best.pt"
    TELEGRAM_TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
    MIN_VEHICLES = 10
    VIDEO_PATH = "video.mp4"

    detector = TrafficDetector(MODEL_PATH, TELEGRAM_TOKEN, MIN_VEHICLES)
    detector.process_video(VIDEO_PATH)