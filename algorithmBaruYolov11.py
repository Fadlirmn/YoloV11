import cv2
import numpy as np
from ultralytics import YOLO
import torch

class TrafficCongestionDetector:
    def __init__(self, model_path, video_source=0, custom_boundaries=None):
        # Inisialisasi model YOLO
        self.model = YOLO(model_path)
        self.video_source = video_source
        
        # Tambahkan custom boundaries
        self.custom_boundaries = custom_boundaries
        
        # Kelas yang akan dideteksi (sesuaikan dengan model YOLO Anda)
        self.vehicle_classes = ['car', 'motorcycle']
        
        # Threshold untuk menentukan tingkat kemacetan
        self.congestion_thresholds = {
            'low': 0.3,    # Kurang dari 30% area terisi
            'medium': 0.6  # Antara 30-60% area terisi
            # Di atas 60% dianggap macet
        }
        
    def calculate_lane_congestion(self, boxes, lane_boundaries, frame_height):
        """
        Menghitung tingkat kemacetan untuk setiap jalur
        """
        lane_vehicles = {0: [], 1: [], 2: []}  # Untuk menyimpan kendaraan per jalur
        
        # Urutkan boxes berdasarkan jarak (y-coordinate)
        sorted_boxes = sorted(boxes, key=lambda x: x[1] + x[3])  # Sort by bottom y-coordinate
        
        for box in sorted_boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            
            # Tentukan jalur berdasarkan posisi x
            for lane_idx, (left, right) in enumerate(lane_boundaries):
                if left <= center_x <= right:
                    lane_vehicles[lane_idx].append(box)
                    break
        
        lane_congestion = {}
        for lane_idx, vehicles in lane_vehicles.items():
            if not vehicles:
                lane_congestion[lane_idx] = {
                    'level': 'clear',
                    'percentage': 0,
                    'vehicle_count': 0
                }
                continue
            
            # Hitung area yang terisi kendaraan
            total_area = (lane_boundaries[lane_idx][1] - lane_boundaries[lane_idx][0]) * frame_height
            vehicle_area = sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in vehicles)
            congestion_percentage = vehicle_area / total_area
            
            # Tentukan level kemacetan
            if congestion_percentage < self.congestion_thresholds['low']:
                level = 'clear'
            elif congestion_percentage < self.congestion_thresholds['medium']:
                level = 'moderate'
            else:
                level = 'congested'
                
            lane_congestion[lane_idx] = {
                'level': level,
                'percentage': congestion_percentage * 100,
                'vehicle_count': len(vehicles)
            }
            
        return lane_congestion

    def get_percentage_boundaries(self, width, percentages):
        """
        Membuat boundaries berdasarkan persentase dari lebar frame
        """
        boundaries = []
        current_x = 0
        for percentage in percentages:
            next_x = int(current_x + (width * percentage / 100))
            boundaries.append((current_x, next_x))
            current_x = next_x
        return boundaries
    
    def process_frame(self, frame):
        """
        Memproses frame dan mendeteksi kemacetan
        """
        height, width = frame.shape[:2]
        
        # Gunakan custom boundaries jika ada, jika tidak bagi rata menjadi 3
        if self.custom_boundaries:
            if isinstance(self.custom_boundaries[0], (int, float)):
                # Jika input adalah persentase
                lane_boundaries = self.get_percentage_boundaries(width, self.custom_boundaries)
            else:
                # Jika input adalah koordinat pixel
                lane_boundaries = self.custom_boundaries
        else:
            # Pembagian default menjadi 3 jalur sama besar
            lane_width = width // 3
            lane_boundaries = [
                (0, lane_width),
                (lane_width, lane_width*2),
                (lane_width*2, width)
            ]
        
        # Deteksi objek menggunakan YOLO
        results = self.model(frame)[0]
        boxes = []
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            if conf > 0.3 and int(cls) in range(len(self.vehicle_classes)):
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        # Hitung kemacetan per jalur
        congestion_status = self.calculate_lane_congestion(boxes, lane_boundaries, height)
        
        # Visualisasi
        for lane_idx, (left, right) in enumerate(lane_boundaries):
            status = congestion_status[lane_idx]
            
            # Warna berdasarkan level kemacetan
            color = {
                'clear': (0, 255, 0),      # Hijau
                'moderate': (0, 255, 255),  # Kuning
                'congested': (0, 0, 255)    # Merah
            }[status['level']]
            
            # Gambar garis pembatas jalur
            cv2.line(frame, (left, 0), (left, height), (255, 255, 255), 2)
            
            # Tampilkan informasi kemacetan
            text = f"Lane {lane_idx+1}: {status['level'].upper()}"
            cv2.putText(frame, text, (left + 10, 30 + lane_idx*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Gambar bounding box
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        return frame, congestion_status
    
    def run(self):
        """
        Menjalankan deteksi kemacetan pada video stream
        """
        cap = cv2.VideoCapture(self.video_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, congestion_status = self.process_frame(frame)
            
            # Tampilkan frame
            cv2.imshow('Traffic Congestion Detection', processed_frame)
            
            # Cetak status kemacetan
            print("\nCongestion Status:")
            for lane_idx, status in congestion_status.items():
                print(f"Lane {lane_idx+1}:")
                print(f"  Level: {status['level']}")
                print(f"  Congestion: {status['percentage']:.1f}%")
                print(f"  Vehicles: {status['vehicle_count']}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Contoh penggunaan:
if __name__ == "__main__":
    # Contoh 1: Pembagian berdasarkan pixel
    custom_boundaries_pixel = [
        (0, 200),           # Jalur 1 dengan lebar 200 pixel
        (200, 500),         # Jalur 2 dengan lebar 300 pixel
        (500, 800)          # Jalur 3 dengan lebar 300 pixel
    ]

    # Contoh 2: Pembagian berdasarkan persentase (30%, 40%, 30%)
    custom_boundaries_percentage = [30, 40, 30]

    # Pilih salah satu jenis custom boundaries
    detector = TrafficCongestionDetector(
        model_path='best.pt',  # Ganti dengan path model Anda
        video_source='video.mp4',  # Ganti dengan path video atau nomor kamera
        custom_boundaries=custom_boundaries_pixel  # atau custom_boundaries_percentage
    )
    
    detector.run()