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
        # ... (bagian lain tetap sama)

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
        # ... (bagian lain tetap sama)

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

    # ... (bagian lain tetap sama)

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