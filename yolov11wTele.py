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
from telebot.handler_backends import State, StatesGroup
from telebot.storage import StateMemoryStorage

class BotStates(StatesGroup):
    monitoring = State()
    configuring = State()

class MultiprocessTrafficDetector:
    def __init__(self, model_path: str, telegram_token: str, min_vehicles: int = 10):
        """
        Initialize the multiprocessing traffic detector with Telegram group support
        
        Args:
            model_path: Path to YOLOv8 model
            telegram_token: Telegram bot token
            min_vehicles: Minimum vehicles to trigger traffic jam alert
        """
        self.model_path = model_path
        self.min_vehicles = min_vehicles
        self.notification_cooldown = 300  # 5 minutes between notifications
        self.last_notification_time = 0
        
        # Initialize bot with state storage
        state_storage = StateMemoryStorage()
        self.bot = telebot.TeleBot(telegram_token, state_storage=state_storage)
        self.active_groups = set()
        
        # Register message handlers
        self.setup_handlers()

    def setup_handlers(self):
        """Set up Telegram message handlers"""
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
            if message.chat.id in self.active_groups:
                status_message = (
                    "✅ Monitoring active\n"
                    f"Minimum vehicles threshold: {self.min_vehicles}\n"
                    f"Notification cooldown: {self.notification_cooldown/60} minutes"
                )
            else:
                status_message = "❌ Monitoring inactive"
            self.bot.reply_to(message, status_message)

        @self.bot.message_handler(commands=['set_min_vehicles'])
        def set_min_vehicles(message):
            try:
                new_threshold = int(message.text.split()[1])
                if new_threshold > 0:
                    self.min_vehicles = new_threshold
                    self.bot.reply_to(
                        message, 
                        f"✅ Minimum vehicles threshold updated to {new_threshold}"
                    )
                else:
                    self.bot.reply_to(message, "❌ Please provide a positive number")
            except (IndexError, ValueError):
                self.bot.reply_to(
                    message, 
                    "❌ Invalid format. Use: /set_min_vehicles [number]"
                )

    def init_worker(self):
        """Initialize YOLO model for each worker process"""
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

        frame = cv2.resize(frame, (640, 480))
        results = model(frame, stream=True)
        
        for result in results:
            vehicle_count = self.count_vehicles(result)
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

            result_queue.put((frame_number, annotated_frame, vehicle_count))

    def send_telegram_notification(self, frame: np.ndarray, vehicle_count: int):
        """Send notification to all active Telegram groups"""
        current_time = time.time()
        
        if current_time - self.last_notification_time < self.notification_cooldown:
            return

        temp_image = f"traffic_jam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_image, frame)

        message = (
            f"🚨 Traffic Jam Detected!\n"
            f"📍 Vehicle Count: {vehicle_count}\n"
            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        with open(temp_image, 'rb') as photo:
            for group_id in self.active_groups:
                try:
                    self.bot.send_photo(group_id, photo, caption=message)
                except Exception as e:
                    print(f"Error sending notification to group {group_id}: {e}")
        
        self.last_notification_time = current_time
        os.remove(temp_image)

    def process_video(self, video_path: str):
        """Process video file using multiple processes"""
        # Start bot polling in a separate process
        bot_process = mp.Process(target=self.bot.polling, args=(True,))
        bot_process.start()

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
        result_queue = mp.Manager().Queue()

        frame_number = 0
        batch_size = 5
        processing_tasks = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % batch_size != 0:
                continue

            task = pool.apply_async(
                self.process_frame, 
                args=((frame, frame_number, total_frames, result_queue),)
            )
            processing_tasks.append(task)

        for task in processing_tasks:
            task.get()

        results = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except Empty:
                break

        for frame_num, annotated_frame, vehicle_count in sorted(results):
            if vehicle_count >= self.min_vehicles:
                self.send_telegram_notification(annotated_frame, vehicle_count)

        pool.close()
        pool.join()
        cap.release()
        cv2.destroyAllWindows()

def main():
    MODEL_PATH = "best.pt"
    TELEGRAM_TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
    MIN_VEHICLES = 10
    VIDEO_PATH = "video.mp4"

    detector = MultiprocessTrafficDetector(MODEL_PATH, TELEGRAM_TOKEN, MIN_VEHICLES)
    detector.process_video(VIDEO_PATH)

if __name__ == "__main__":
    main()