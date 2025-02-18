import cv2
import numpy as np
from ultralytics import YOLO
import torch

class TrafficCongestionDetector:
    def __init__(self, model_path, video_source=0, lane_points=None):
        self.model = YOLO(model_path)
        self.video_source = video_source
        
        # Format: [[(x1_start,y1_start), (x1_end,y1_end)], [(x2_start,y2_start), (x2_end,y2_end)], ...]
        self.lane_points = lane_points
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
        self.congestion_thresholds = {'low': 0.3, 'medium': 0.6}

    def point_to_line_distance(self, point, line_start, line_end):
        """Menghitung jarak dari titik ke garis"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        return numerator/denominator

    def determine_lane(self, point, lane_lines):
        """Menentukan jalur untuk suatu titik berdasarkan jarak ke garis pembatas"""
        distances = []
        for line in lane_lines:
            dist = self.point_to_line_distance(point, line[0], line[1])
            distances.append(dist)
        
        # Titik berada di jalur dengan jarak minimum ke garis pembatas
        return np.argmin(distances)

    def calculate_lane_congestion(self, boxes, frame_height, frame_width):
        lane_vehicles = {0: [], 1: [], 2: []}
        
        # Urutkan boxes berdasarkan jarak
        sorted_boxes = sorted(boxes, key=lambda x: x[1] + x[3])
        
        for box in sorted_boxes:
            x1, y1, x2, y2 = box
            center = ((x1 + x2)/2, (y1 + y2)/2)
            
            # Tentukan jalur berdasarkan jarak ke garis pembatas
            lane_idx = self.determine_lane(center, self.lane_points)
            lane_vehicles[lane_idx].append(box)
        
        lane_congestion = {}
        for lane_idx in lane_vehicles:
            vehicles = lane_vehicles[lane_idx]
            if not vehicles:
                lane_congestion[lane_idx] = {
                    'level': 'clear',
                    'percentage': 0,
                    'vehicle_count': 0
                }
                continue
            
            # Hitung area yang terisi relatif terhadap area jalur
            lane_area = frame_height * frame_width / 3  # Perkiraan kasar area jalur
            vehicle_area = sum((x2-x1)*(y2-y1) for x1,y1,x2,y2 in vehicles)
            congestion_percentage = vehicle_area / lane_area
            
            level = 'congested'
            if congestion_percentage < self.congestion_thresholds['low']:
                level = 'clear'
            elif congestion_percentage < self.congestion_thresholds['medium']:
                level = 'moderate'
            
            lane_congestion[lane_idx] = {
                'level': level,
                'percentage': congestion_percentage * 100,
                'vehicle_count': len(vehicles)
            }
            
        return lane_congestion

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        
        results = self.model(frame)[0]
        boxes = []
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            if conf > 0.3 and int(cls) in range(len(self.vehicle_classes)):
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        congestion_status = self.calculate_lane_congestion(boxes, height, width)
        
        # Visualisasi
        # Gambar garis pembatas miring
        for i, line in enumerate(self.lane_points):
            start_point = tuple(map(int, line[0]))
            end_point = tuple(map(int, line[1]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
        
        # Gambar bounding box dengan warna sesuai kemacetan
        for box in boxes:
            x1, y1, x2, y2 = box
            center = ((x1 + x2)/2, (y1 + y2)/2)
            lane_idx = self.determine_lane(center, self.lane_points)
            
            color = {
                'clear': (0, 255, 0),
                'moderate': (0, 255, 255),
                'congested': (0, 0, 255)
            }[congestion_status[lane_idx]['level']]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Tampilkan status
        for lane_idx, status in congestion_status.items():
            text = f"Lane {lane_idx+1}: {status['level'].upper()}"
            cv2.putText(frame, text, (10, 30 + lane_idx*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
        
        return frame, congestion_status

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, congestion_status = self.process_frame(frame)
            cv2.imshow('Traffic Congestion Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Contoh penggunaan dengan jalur miring
if __name__ == "__main__":
    # Definisikan titik-titik untuk jalur miring
    # Format: [(x_start,y_start), (x_end,y_end)]
    angled_lanes = [
        [(100,0), (0,480)],    # Garis pembatas kiri
        [(320,0), (240,480)],  # Garis pembatas tengah
        [(540,0), (480,480)]   # Garis pembatas kanan
    ]
    
    detector = TrafficCongestionDetector(
        model_path='best.pt',
        video_source='video.mp4',
        lane_points=angled_lanes
    )
    detector.run()