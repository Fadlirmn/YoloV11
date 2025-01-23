import cv2
import numpy as np
from ultralytics import YOLO
import telebot
from datetime import datetime
import time
import os
import multiprocessing as mp
from threading import Thread
import RPi.GPIO as GPIO  # Added for GPIO rain sensor support

def setup_gpio():
    """Initialize GPIO settings for Raspberry Pi"""
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Rain sensor pin setup
        rain_pin = 17
        GPIO.setup(rain_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        return rain_pin
    except Exception as e:
        print(f"GPIO Setup Error: {e}")
        return None

class TelegramBot:
    def __init__(self, token: str, min_vehicles: int = 10):
        # GPIO Setup
        self.rain_pin = setup_gpio()
        
        self.bot = telebot.TeleBot(token)
        self.active_groups = set()
        self.min_vehicles = min_vehicles
        self.last_vehicle_count = 0
        self.last_detection_time = None
        self.latest_frame_path = None
        self.is_rain = False
        self.is_running = True
        
        self.setup_handlers()

    def update_status(self, vehicle_count: int, image_path: str):
        """Update bot's internal status with latest detection information"""
        self.last_vehicle_count = vehicle_count
        self.last_detection_time = datetime.now()
        self.latest_frame_path = image_path

        try:
            # Check Rain Sensor Status
            self.is_rain = GPIO.input(self.rain_pin) == GPIO.LOW
        except Exception as e:
            print(f"Rain sensor error: {e}")
            self.is_rain = False

    def setup_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def start(message):
            if message.chat.type in ['group', 'supergroup']:
                self.active_groups.add(message.chat.id)
                welcome_message = (
                    "🚦 Traffic Monitoring Bot activated!\n\n"
                    "/start - Start monitoring\n"
                    "/stop - Stop monitoring\n"
                    "/status - Check status\n"
                    "/current - View traffic status\n"
                    "/latest - Show latest detection\n"
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
                    # Split the message and get the second part (index 1)
                    parts = message.text.split()
                    if len(parts) < 2:
                        self.bot.reply_to(message, "❌ Please provide a number")
                        return

                    # Attempt to convert to integer
                    new_threshold = int(parts[1])
                    
                    if new_threshold < 1: 
                        raise ValueError("Threshold must be at least 1")
                    
                    self.min_vehicles = new_threshold
                    self.bot.reply_to(message, f"✅ Threshold set to: {new_threshold}")
                
                except ValueError:
                    self.bot.reply_to(message, "❌ Invalid number. Please enter a valid integer.")
                except Exception as e:
                    self.bot.reply_to(message, f"❌ Error: {str(e)}")

        @self.bot.message_handler(commands=['rain_status'])
        def rain_status(message):
            rain_message = "🌧️ Rain Detected" if self.is_rain else "☀️ No Rain"
            self.bot.reply_to(message, rain_message)

    def send_notification(self, image_path: str, vehicle_count: int):
        self.update_status(vehicle_count, image_path)
        rain_status = "🌧️ Rain Detected" if self.is_rain else "☀️ No Rain"
        message = (
            f"🚨 Traffic Jam!\n"
            f"📍 Vehicles: {vehicle_count}\n"
            f"🌈 Weather: {rain_status}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        for group_id in self.active_groups:
            try:
                with open(image_path, 'rb') as photo:
                    self.bot.send_photo(group_id, photo, caption=message)
            except Exception as e:
                print(f"Error sending to group {group_id}: {e}")

    def start_polling(self):
        self.bot.polling(none_stop=True)

def process_frame(frame, frame_number: int, model_path: str) -> tuple:
    try:
        model = YOLO(model_path)
        frame = cv2.resize(frame, (640, 480))
        results = model(frame, stream=True)
        
        for result in results:
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
            vehicle_count = sum(1 for box in result.boxes if result.names[int(box.cls)] in vehicle_classes)
            annotated_frame = result.plot()
            cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            temp_image = f"frame_{frame_number}.jpg"
            cv2.imwrite(temp_image, annotated_frame)
            return temp_image, vehicle_count
        
        return None, 0
    except Exception as e:
        print(f"Frame processing error: {e}")
        return None, 0

class TrafficDetector:
    def __init__(self, model_path: str, telegram_token: str, min_vehicles: int = 10):
        self.model_path = model_path
        self.min_vehicles = min_vehicles
        self.notification_cooldown = 30  # 30s
        self.last_notification_time = 0
        self.is_running = True
        
        self.telegram_bot = TelegramBot(telegram_token, min_vehicles)
        self.bot_thread = Thread(target=self.telegram_bot.start_polling)
        self.bot_thread.daemon = True
        self.bot_thread.start()

    def process_video(self, video_path: str):
        while self.is_running:
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error opening video: {video_path}")
                    break

                frame_number = 0
                
                while cap.isOpened() and self.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_number += 1
                    if frame_number % 5 != 0:  # Process every 5th frame
                        continue

                    temp_image_path, vehicle_count = process_frame(frame, frame_number, self.model_path)
                    if temp_image_path:
                        self.telegram_bot.update_status(vehicle_count, temp_image_path)
                        
                        current_time = time.time()
                        if (vehicle_count >= self.min_vehicles and 
                            current_time - self.last_notification_time >= self.notification_cooldown):
                            self.telegram_bot.send_notification(temp_image_path, vehicle_count)
                            self.last_notification_time = current_time

                cap.release()
                print("Video ended, restarting...")
                time.sleep(1)  # Wait before restarting
            
            except Exception as e:
                print(f"Video processing error: {e}")
                time.sleep(5)  # Wait before retry

    def stop(self):
        self.is_running = False
        try:
            GPIO.cleanup()  # Clean up GPIO
        except:
            pass

def main():
    MODEL_PATH = "best.pt"
    TELEGRAM_TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
    MIN_VEHICLES = 8
    VIDEO_PATH = "video.mp4"

    # Ensure GPIO is set up before starting
    setup_gpio()

    detector = TrafficDetector(MODEL_PATH, TELEGRAM_TOKEN, MIN_VEHICLES)
    try:
        detector.process_video(VIDEO_PATH)
    except KeyboardInterrupt:
        print("Stopping...")
        detector.stop()
    finally:
        GPIO.cleanup()  # Ensure GPIO is cleaned up

if __name__ == "__main__":
    main()