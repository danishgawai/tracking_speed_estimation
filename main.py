__author__ = "Danish Ahmed"
__copyright__ = ""
__credits__ = ["Danish Ahmed"]
__license__ = ""
__version__ = "0.2.0" 
__maintainer__ = "Danish Ahmed Gawai"
__email__ = "danishh163@gmail.com"
__status__ = "Development"
__module_name__ = "Main"


import sys
import logging
import cv2
import time
import numpy as np
import traceback
from collections import defaultdict
from datetime import datetime
import json

# Import the new vector speed estimator
from utils.vector_speed_estimator import VectorSpeedEstimator
from utils.yolov8_infer import YOLOv8Inference
from tracker.byte_tracker import BYTETracker

# Configuration
source_stream = "aerial4.mp4"
maxDisappeared = 20
ROI = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]

# Initialize detector
detector = YOLOv8Inference(
    model_path="models/yolov8s_merger8_exp1.pt",
    conf_thres=0.5,
)

logging.info("Starting processing with Vector-based Speed Estimation")

def plot_detections(frame, detections):
    """Enhanced detection plotting"""
    if detections:
        for (x1, y1, x2, y2, conf_percent, cls_id) in detections:
            label = f"ClsID {cls_id}: {conf_percent}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 100), 2)
            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_bg_y1 = max(y1 - text_h - baseline - 3, 0)
            text_bg_y2 = y1 - baseline + 3
            cv2.rectangle(frame, (x1, text_bg_y1), (x1 + text_w + 4, text_bg_y2), (200, 0, 100), -1)
            cv2.putText(frame, label, (x1 + 2, text_bg_y2 - baseline - 3), font, font_scale, (255, 255, 255), thickness)
    return frame

def draw_track_info(frame, track_data, position):
    """Draw comprehensive track information on frame"""
    y_offset = position[1]
    
    for track_id, data in track_data.items():
        if not data:
            continue
            
        # Format track information
        info_lines = [
            f"Track {track_id}:",
            f"Speed: {data.get('speed_kmh', 0):.1f} km/h",
            f"Direction: {data.get('direction_degrees', 0):.0f}°",
            f"Confidence: {data.get('confidence', 0):.2f}",
            ""
        ]
        
        for line in info_lines:
            cv2.putText(frame, line, (position[0], y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
            
        if y_offset > frame.shape[0] - 50:  # Reset if too low
            y_offset = 30

def save_speed_data(track_data, frame_count, output_file="speed_data.json"):
    """Save speed data to JSON for analysis"""
    timestamp = time.time()
    
    frame_data = {
        "frame": frame_count,
        "timestamp": timestamp,
        "tracks": {}
    }
    
    for track_id, data in track_data.items():
        if data:
            frame_data["tracks"][str(track_id)] = {
                "speed_ms": data.get('speed_magnitude', 0),
                "speed_kmh": data.get('speed_kmh', 0),
                "direction_deg": data.get('direction_degrees', 0),
                "velocity_vector": data.get('velocity_vector', [0, 0]),
                "confidence": data.get('confidence', 0)
            }
    
    # Append to file
    try:
        with open(output_file, 'a') as f:
            f.write(json.dumps(frame_data) + "\n")
    except Exception as e:
        logging.warning(f"Could not save speed data: {e}")

def predict_on_RTSP():
    cap = cv2.VideoCapture(source_stream)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {source_stream}")
        return
    
    time_string = str(datetime.now()).replace(" ", "_").replace(":", "-")
    output_video_path = f"video_out_vector_speed_{time_string}.mp4"
    speed_data_file = f"speed_data_{time_string}.json"
    
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    # Initialize tracker
    tracker = BYTETracker(
        track_thresh=0.5,
        track_buffer=maxDisappeared,
        match_thresh=0.8,
        frame_rate=float(fps),
    )
    
    # Initialize Vector Speed Estimator
    speed_estimator = VectorSpeedEstimator(
        fps=fps,
        homography_update_interval=5,
        smoothing_window=5,
        motion_compensation_method="homography"  # Can be changed to "optical_flow" or "hybrid"
    )
    
    # Update GSD if you know drone parameters
    # speed_estimator.update_gsd(altitude_m=100, focal_length_mm=4.5)  # Uncomment and adjust
    
    frame_count, total_dets, total_time = 0, 0, 0.0
    track_history = defaultdict(list)
    track_speed_data = {}
    run_time = 10000  # seconds
    initialize_time = time.time()
    
    print(f"Processing with Vector-based Speed Estimation")
    print(f"Motion Compensation: {speed_estimator.motion_compensation_method}")
    print(f"Output Video: {output_video_path}")
    print(f"Speed Data: {speed_data_file}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        loop_start_time = time.time()
        frame_count += 1
        annotated_frame = frame.copy()
        detections = []

        # Update drone motion compensation
        try:
            speed_estimator.update_drone_motion(frame)
        except Exception as e:
            logging.warning(f"Motion compensation failed: {e}")

        # Detection
        try:
            boxes = detector.infer(annotated_frame)
            # Convert to detection format if needed
            for box in boxes:
                if len(box) >= 6:  # [x1, y1, x2, y2, conf, cls]
                    x1, y1, x2, y2, conf, cls_id = box[:6]
                    detections.append((int(x1), int(y1), int(x2), int(y2), 
                                     round(conf * 100, 1), int(cls_id)))
        except Exception as e:
            logging.critical(f"{__module_name__} Detector failed. Error: {e}")
            logging.debug(f"{__module_name__}:\n {traceback.format_exc()}")

        # Tracking
        try:
            if len(boxes) > 0:
                centroids_boxes = tracker.update(np.array(boxes).astype(np.float32))
            else:
                centroids_boxes = []
        except Exception as e:
            logging.critical(f"{__module_name__} Error in Tracking: {e}")
            logging.debug(f'{__module_name__}: \n {traceback.format_exc()}')
            time.sleep(0.1)
            continue

        # Vector-based Speed Estimation
        current_timestamp = time.time()
        track_speed_data = {}
        
        for centroid_id, centroid, box, score, _ in centroids_boxes:
            box = box.astype(np.int32)
            
            # Calculate vector speed
            try:
                speed_data = speed_estimator.calculate_vector_speed(
                    track_id=centroid_id,
                    bbox=[int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    timestamp=current_timestamp
                )
                
                if speed_data:
                    track_speed_data[centroid_id] = speed_data
                
            except Exception as e:
                logging.warning(f"Speed estimation failed for track {centroid_id}: {e}")
            
            # Draw bounding box and track ID
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"ID: {centroid_id}", 
                       (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update track history for trajectory visualization
            track = track_history[centroid_id]
            track.append((int(centroid[0]), int(centroid[1])))
            if len(track) > 30:
                track.pop(0)
            
            # Draw trajectory
            if len(track) > 1:
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        # Visualize velocity vectors
        annotated_frame = speed_estimator.visualize_vectors(annotated_frame, track_speed_data)

        # Draw comprehensive track information
        draw_track_info(annotated_frame, track_speed_data, position=(10, 30))

        # Detect and display anomalies
        try:
            anomalies = speed_estimator.detect_anomalies(track_speed_data)
            if anomalies:
                anomaly_y = height - 150
                cv2.putText(annotated_frame, "ANOMALIES DETECTED:", 
                           (10, anomaly_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                anomaly_y += 25
                
                for anomaly in anomalies[:5]:  # Show max 5 anomalies
                    text = f"Track {anomaly['track_id']}: {anomaly['description']}"
                    cv2.putText(annotated_frame, text, (10, anomaly_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    anomaly_y += 20
        except Exception as e:
            logging.warning(f"Anomaly detection failed: {e}")

        # Draw motion compensation info
        comp_info = [
            f"Frame: {frame_count}",
            f"Method: {speed_estimator.motion_compensation_method}",
            f"Drone Motion: [{speed_estimator.drone_velocity[0]:.2f}, {speed_estimator.drone_velocity[1]:.2f}] px/frame",
            f"GSD: {speed_estimator.gsd:.4f} m/px",
            f"Active Tracks: {len(track_speed_data)}"
        ]
        
        for i, info in enumerate(comp_info):
            cv2.putText(annotated_frame, info, (width - 400, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Overlay original detections
        annotated_frame = plot_detections(annotated_frame, detections)
        
        # Save speed data
        save_speed_data(track_speed_data, frame_count, speed_data_file)
        
        # Write frame
        out.write(annotated_frame)

        # Performance statistics
        num_dets_frame = len(detections)
        total_dets += num_dets_frame
        loop_time = time.time() - loop_start_time
        total_time += loop_time
        avg_fps_overall = frame_count / total_time if total_time > 0 else float('inf')
        
        # Enhanced logging
        active_speeds = [data.get('speed_kmh', 0) for data in track_speed_data.values() if data]
        avg_speed = np.mean(active_speeds) if active_speeds else 0
        
        print(f"Frame: {frame_count}, Dets: {num_dets_frame}, Tracks: {len(track_speed_data)}, "
              f"Avg Speed: {avg_speed:.1f} km/h, Time: {loop_time*1000:.2f}ms, FPS: {avg_fps_overall:.2f}")

        if (time.time() - initialize_time) > run_time:
            print("Stopping, saving video and data...")
            break

    # Generate final statistics
    try:
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        
        all_track_stats = {}
        for track_id in speed_estimator.tracks.keys():
            stats = speed_estimator.get_track_statistics(track_id)
            if stats:
                all_track_stats[track_id] = stats
                print(f"Track {track_id}:")
                print(f"  Avg Speed: {stats['avg_speed_kmh']:.1f} km/h")
                print(f"  Max Speed: {stats['max_speed_kmh']:.1f} km/h")
                print(f"  Direction Consistency: {stats['direction_consistency']:.2f}")
                print(f"  Total Distance: {stats['total_distance']:.1f} m")
                print(f"  Duration: {stats['track_duration']:.1f} s")
                print()
        
        # Save final statistics
        stats_file = f"final_statistics_{time_string}.json"
        with open(stats_file, 'w') as f:
            json.dump(all_track_stats, f, indent=2)
        print(f"Final statistics saved to: {stats_file}")
        
    except Exception as e:
        logging.warning(f"Could not generate final statistics: {e}")

    cap.release()
    out.release()
    print(f"\nProcessing completed!")
    print(f"Output video: {output_video_path}")
    print(f"Speed data: {speed_data_file}")

if __name__ == "__main__":
    print("Starting Vector-based Speed Estimation Processing")
    try:
        predict_on_RTSP()
    except KeyboardInterrupt:
        print("Interrupted by user, saving video...")
        sys.exit(0)
    except Exception as e:
        print(f"Code crashed unexpectedly: {e}")
        print(traceback.format_exc())
    print("Processing Completed with Vector-based Speed Estimation")