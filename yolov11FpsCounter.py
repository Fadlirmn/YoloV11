import cv2
import numpy as np
from ultralytics import YOLO
import telebot
from datetime import datetime
import time
import os
import io
from threading import Thread

def process_frame(frame, frame_number: int, model_path: str) -> tuple:
    start_time = time.time()  # Waktu mulai pemrosesan frame
    
    model = YOLO(model_path)
    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)
    
    for result in results:
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        vehicle_count = sum(1 for box in result.boxes if result.names[int(box.cls)] in vehicle_classes)
        annotated_frame = result.plot()
        
        # Hitung FPS
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 1 / processing_time if processing_time > 0 else 0
        
        # Tambahkan teks FPS ke frame
        cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Konversi frame ke bytes untuk pengiriman
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = io.BytesIO(buffer)

        return frame_bytes, vehicle_count, fps
    
    return None, 0, 0

class TelegramBot:
    def __init__(self, token: str, min_vehicles: int = 10):
        self.bot = telebot.TeleBot(token)
        self.active_groups = set()
        self.min_vehicles = min_vehicles
        self.last_vehicle_count = 0
        self.last_detection_time = None
        self.latest_frame_path = None
        self.is_rain = False
        self.is_running = True
        self.setup_handlers()

    def setup_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def start(message):
            if message.chat.type in ['group', 'supergroup']:
                self.active_groups.add(message.chat.id)
                welcome_message = (
                    "🚦 Traffic Monitoring Bot activated!\n\n"
                    "/start - Start monitoring\n"
                    "/stop - Stop monitoring\n"
                    "/status - Check monitoring status\n"
                    "/current - View current traffic status\n"
                    "/latest - Show latest detection\n"
                    "/help - Show all commands\n"
                    "/set_min_vehicles [number] - Set threshold\n"
                    "/rain_status - Check current rain status"
                )
                self.bot.reply_to(message, welcome_message)

        @self.bot.message_handler(commands=['stop'])
        def stop(message):
            if message.chat.id in self.active_groups:
                self.active_groups.remove(message.chat.id)
                self.bot.reply_to(message, "❌ Monitoring stopped.")

        @self.bot.message_handler(commands=['status'])
        def status(message):
            status_message = f"✅ Monitoring active\n🚗 Threshold: {self.min_vehicles}" if message.chat.id in self.active_groups else "❌ Inactive"
            self.bot.reply_to(message, status_message)

        @self.bot.message_handler(commands=['current'])
        def current(message):
            if message.chat.id not in self.active_groups:
                self.bot.reply_to(message, "❌ Not active.")
                return

            traffic_status = "🔴 Heavy" if self.last_vehicle_count >= self.min_vehicles else "🟢 Normal"
            rain_status = "🌧️ Rain Detected" if self.is_rain else "☀️ No Rain"
            detection_time = self.last_detection_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_detection_time else "No detection"
            
            status_message = (
                f"🚦 Status: {traffic_status}\n"
                f"🌈 Weather: {rain_status}\n"
                f"🚗 Vehicles: {self.last_vehicle_count}\n"
                f"⚠️ Threshold: {self.min_vehicles}\n"
                f"⏰ Updated: {detection_time}"
            )
            self.bot.reply_to(message, status_message)

        @self.bot.message_handler(commands=['latest'])
        def latest(message):
            if message.chat.id not in self.active_groups or not self.latest_frame_path or not os.path.exists(self.latest_frame_path):
                self.bot.reply_to(message, "No data available.")
                return

            detection_time = self.last_detection_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_detection_time else "No detection"
            rain_status = "🌧️ Rain Detected" if self.is_rain else "☀️ No Rain"
            status_message = (
                f"🚗 Vehicles: {self.last_vehicle_count}\n"
                f"🌈 Weather: {rain_status}\n"
                f"⏰ Time: {detection_time}"
            )
            
            try:
                with open(self.latest_frame_path, 'rb') as photo:
                    self.bot.send_photo(message.chat.id, photo, caption=status_message)
            except Exception as e:
                self.bot.reply_to(message, f"Error: {e}")

        @self.bot.message_handler(commands=['set_min_vehicles'])
        def set_min_vehicles(message):
            try:
                new_threshold = int(message.text.split()[1])
                if new_threshold < 1: raise ValueError
                self.min_vehicles = new_threshold
                self.bot.reply_to(message, f"✅ Threshold: {new_threshold}")
            except:
                self.bot.reply_to(message, "❌ Invalid number")

        @self.bot.message_handler(commands=['rain_status'])
        def rain_status(message):
            rain_message = "🌧️ Rain Detected" if self.is_rain else "☀️ No Rain"
            self.bot.reply_to(message, rain_message)


    def send_notification(self, frame_bytes: io.BytesIO, vehicle_count: int, custom_message: str = None):
        self.update_status(vehicle_count, None)
        message = custom_message or f"🚨 Traffic Jam!\n📍 Vehicles: {vehicle_count}\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        for group_id in self.active_groups:
            try:
                self.bot.send_photo(group_id, frame_bytes, caption=message)
            except Exception as e:
                print(f"Error sending to group {group_id}: {e}")

class TrafficDetector:
    def __init__(self, model_path: str, telegram_token: str, min_vehicles: int = 10):
        self.model_path = model_path
        self.min_vehicles = min_vehicles
        self.notification_cooldown = 300
        self.last_notification_time = 0
        self.is_running = True
        
        self.telegram_bot = TelegramBot(telegram_token, min_vehicles)
        self.bot_thread = Thread(target=self.telegram_bot.start_polling)
        self.bot_thread.daemon = True
        self.bot_thread.start()

    def process_video(self, video_path: str):
        while self.is_running:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video: {video_path}")
                return

            frame_number = 0
            
            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                if frame_number % 5 != 0:  # Proses setiap 5 frame
                    continue

                try:
                    frame_bytes, vehicle_count, fps = process_frame(frame, frame_number, self.model_path)
                    if frame_bytes:
                        # Pembaruan statistik FPS
                        self.total_fps += fps
                        self.processed_frames += 1
                        self.max_fps = max(self.max_fps, fps)
                        self.min_fps = min(self.min_fps, fps)
                        
                        # Rata-rata FPS
                        avg_fps = self.total_fps / self.processed_frames if self.processed_frames > 0 else 0
                        
                        # Logging statistik FPS
                        print(f"Frame {frame_number}:")
                        print(f"  Current FPS: {fps:.2f}")
                        print(f"  Average FPS: {avg_fps:.2f}")
                        print(f"  Max FPS: {self.max_fps:.2f}")
                        print(f"  Min FPS: {self.min_fps:.2f}")
                        
                        current_time = time.time()
                        if (vehicle_count >= self.min_vehicles and 
                            current_time - self.last_notification_time >= self.notification_cooldown):
                            # Tambahkan informasi FPS ke notifikasi
                            notification_message = (
                                f"🚨 Traffic Jam!\n"
                                f"📍 Vehicles: {vehicle_count}\n"
                                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"📊 Performance:\n"
                                f"   Current FPS: {fps:.2f}\n"
                                f"   Avg FPS: {avg_fps:.2f}"
                            )
                            self.telegram_bot.send_notification(frame_bytes, vehicle_count, notification_message)
                            self.last_notification_time = current_time
                            
                except Exception as e:
                    print(f"Error processing frame {frame_number}: {e}")

            cap.release()
            print("Video ended, restarting...")
            time.sleep(1)  # Wait before restarting

    def stop(self):
        self.is_running = False

if __name__ == "__main__":
    MODEL_PATH = "best.pt"
    TELEGRAM_TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
    MIN_VEHICLES = 10
    VIDEO_PATH = "video.mp4"

    detector = TrafficDetector(MODEL_PATH, TELEGRAM_TOKEN, MIN_VEHICLES)
    try:
        detector.process_video(VIDEO_PATH)
    except KeyboardInterrupt:
        print("Stopping...")
        detector.stop()