# Vector Speed Estimator for Drone-based Vehicle Tracking
# Handles drone motion compensation using multiple methods

import numpy as np
import cv2
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Union
import logging

class VectorSpeedEstimator:
    """
    Advanced vector-based speed estimation with comprehensive drone motion compensation
    """
    
    def __init__(self, 
                 fps: float = 30,
                 homography_update_interval: int = 5,
                 smoothing_window: int = 5,
                 motion_compensation_method: str = "homography"):
        """
        Initialize Vector Speed Estimator
        
        Args:
            fps: Video frame rate
            homography_update_interval: Frames between homography recalculation
            smoothing_window: Number of frames for velocity smoothing
            motion_compensation_method: "homography", "optical_flow", or "hybrid"
        """
        self.fps = fps
        self.homography_update_interval = homography_update_interval
        self.smoothing_window = smoothing_window
        self.motion_compensation_method = motion_compensation_method
        
        # Track data storage
        self.tracks = defaultdict(lambda: deque(maxlen=smoothing_window * 2))
        self.velocity_history = defaultdict(lambda: deque(maxlen=smoothing_window))
        
        # Motion compensation data
        self.homography_matrix = np.eye(3, dtype=np.float32)
        self.prev_frame_gray = None
        self.frame_count = 0
        self.drone_velocity = np.array([0.0, 0.0])  # [vx, vy] in pixels/frame
        
        # Feature detection for motion compensation
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Kalman filters for individual tracks
        self.kalman_filters = {}
        
        # Ground Sampling Distance (GSD) - meters per pixel
        self.gsd = 0.1  # Default, should be calibrated based on altitude
        
        logging.info(f"VectorSpeedEstimator initialized with {motion_compensation_method} compensation")
    
    def update_gsd(self, altitude_m: float, focal_length_mm: float = 4.5, 
                   sensor_width_mm: float = 6.17, image_width_px: int = 1920):
        """
        Update Ground Sampling Distance based on drone parameters
        
        Args:
            altitude_m: Drone altitude in meters
            focal_length_mm: Camera focal length in mm
            sensor_width_mm: Camera sensor width in mm  
            image_width_px: Image width in pixels
        """
        self.gsd = (altitude_m * sensor_width_mm) / (focal_length_mm * image_width_px)
        logging.info(f"GSD updated to {self.gsd:.4f} m/pixel at altitude {altitude_m}m")
    
    def _detect_and_match_features(self, curr_frame_gray: np.ndarray) -> np.ndarray:
        """
        Detect and match features between consecutive frames for motion estimation
        
        Returns:
            Motion transformation matrix (affine or homography)
        """
        if self.prev_frame_gray is None:
            self.prev_frame_gray = curr_frame_gray.copy()
            return np.eye(3, dtype=np.float32)
        
        # Detect ORB features
        kp1, des1 = self.orb.detectAndCompute(self.prev_frame_gray, None)
        kp2, des2 = self.orb.detectAndCompute(curr_frame_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return np.eye(3, dtype=np.float32)
        
        # Match features
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 20:  # Need sufficient matches
            return np.eye(3, dtype=np.float32)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        
        # Estimate transformation based on method
        try:
            if self.motion_compensation_method == "homography":
                # Full perspective transformation
                H, mask = cv2.findHomography(src_pts, dst_pts, 
                                           cv2.RANSAC, 5.0)
                if H is not None:
                    return H
            else:
                # Affine transformation (faster, less accurate)
                M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
                if M is not None:
                    # Convert to homography format
                    H = np.vstack([M, [0, 0, 1]])
                    return H
        except:
            pass
        
        return np.eye(3, dtype=np.float32)
    
    def _optical_flow_compensation(self, curr_frame_gray: np.ndarray) -> np.ndarray:
        """
        Estimate camera motion using sparse optical flow
        """
        if self.prev_frame_gray is None:
            self.prev_frame_gray = curr_frame_gray.copy()
            return np.array([0.0, 0.0])
        
        # Parameters for corner detection
        feature_params = dict(maxCorners=100, qualityLevel=0.3, 
                            minDistance=7, blockSize=7)
        
        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Detect corners in previous frame
        p0 = cv2.goodFeaturesToTrack(self.prev_frame_gray, mask=None, **feature_params)
        
        if p0 is None or len(p0) < 10:
            return np.array([0.0, 0.0])
        
        # Calculate optical flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame_gray, curr_frame_gray, 
                                                    p0, None, **lk_params)
        
        # Select good points
        if p1 is None:
            return np.array([0.0, 0.0])
        
        good_new = p1[status == 1]
        good_old = p0[status == 1]
        
        if len(good_new) < 5:
            return np.array([0.0, 0.0])
        
        # Calculate median displacement (robust to outliers)
        displacement = good_new - good_old
        median_displacement = np.median(displacement, axis=0)
        
        return median_displacement
    
    def update_drone_motion(self, frame: np.ndarray) -> None:
        """
        Update drone motion estimation using the specified method
        """
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.frame_count % self.homography_update_interval == 0:
            if self.motion_compensation_method in ["homography", "hybrid"]:
                self.homography_matrix = self._detect_and_match_features(curr_frame_gray)
                
                # Extract translation from homography for drone velocity
                if not np.array_equal(self.homography_matrix, np.eye(3)):
                    # Get translation components
                    self.drone_velocity = np.array([
                        self.homography_matrix[0, 2],  # tx
                        self.homography_matrix[1, 2]   # ty
                    ])
            
            elif self.motion_compensation_method == "optical_flow":
                self.drone_velocity = self._optical_flow_compensation(curr_frame_gray)
        
        self.prev_frame_gray = curr_frame_gray.copy()
        self.frame_count += 1
    
    def _create_kalman_filter(self) -> cv2.KalmanFilter:
        """Create Kalman filter for track prediction"""
        kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, vx, vy), 2 measurements (x, y)
        
        # Transition matrix (constant velocity model)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Measurement noise
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance
        kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        
        return kalman
    
    def calculate_vector_speed(self, track_id: int, bbox: List[int], 
                             timestamp: float) -> Optional[Dict]:
        """
        Calculate vector-based speed with comprehensive motion compensation
        
        Args:
            track_id: Unique track identifier
            bbox: Bounding box [x1, y1, x2, y2]
            timestamp: Current timestamp
            
        Returns:
            Dictionary with speed metrics or None
        """
        # Extract center point
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        current_pos = np.array([center_x, center_y])
        
        # Initialize Kalman filter if new track
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = self._create_kalman_filter()
            # Initialize state with first position
            self.kalman_filters[track_id].statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32)
            self.kalman_filters[track_id].statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32)
        
        # Store position with timestamp
        self.tracks[track_id].append((current_pos, timestamp))
        
        if len(self.tracks[track_id]) < 2:
            return None
        
        # Get previous position
        prev_pos, prev_timestamp = self.tracks[track_id][-2]
        
        # Calculate time difference
        dt = timestamp - prev_timestamp
        if dt <= 0:
            return None
        
        # Raw displacement in pixels
        displacement_px = current_pos - prev_pos
        
        # Compensate for drone motion
        compensated_displacement_px = displacement_px - (self.drone_velocity * dt * self.fps)
        
        # Convert to world coordinates using GSD
        displacement_world = compensated_displacement_px * self.gsd  # meters
        
        # Calculate velocity vector
        velocity_vector = displacement_world / dt  # m/s
        
        # Update Kalman filter
        kalman = self.kalman_filters[track_id]
        
        # Predict
        prediction = kalman.predict()
        
        # Update with measurement
        measurement = np.array([center_x, center_y], dtype=np.float32)
        kalman.correct(measurement)
        
        # Get smoothed velocity from Kalman filter
        smoothed_velocity = np.array([kalman.statePost[2], kalman.statePost[3]]) * self.gsd
        
        # Store velocity history for additional smoothing
        self.velocity_history[track_id].append(velocity_vector)
        
        # Calculate smoothed velocity using moving average
        if len(self.velocity_history[track_id]) >= 2:
            recent_velocities = list(self.velocity_history[track_id])
            smoothed_velocity_ma = np.mean(recent_velocities, axis=0)
        else:
            smoothed_velocity_ma = velocity_vector
        
        # Calculate final metrics
        speed_magnitude = np.linalg.norm(smoothed_velocity_ma)
        direction_radians = np.arctan2(smoothed_velocity_ma[1], smoothed_velocity_ma[0])
        direction_degrees = np.degrees(direction_radians) % 360
        
        return {
            'track_id': track_id,
            'velocity_vector': smoothed_velocity_ma.tolist(),  # [vx, vy] m/s
            'speed_magnitude': float(speed_magnitude),         # |v| m/s
            'speed_kmh': float(speed_magnitude * 3.6),        # km/h
            'direction_radians': float(direction_radians),     # radians
            'direction_degrees': float(direction_degrees),     # degrees
            'displacement_world': displacement_world.tolist(), # [dx, dy] meters
            'displacement_pixels': displacement_px.tolist(),   # [dx, dy] pixels
            'drone_compensation': self.drone_velocity.tolist(), # [vx, vy] pixels/frame
            'time_interval': float(dt),                        # seconds
            'confidence': self._calculate_confidence(track_id)
        }
    
    def _calculate_confidence(self, track_id: int) -> float:
        """Calculate confidence score based on track stability"""
        if len(self.velocity_history[track_id]) < 3:
            return 0.5
        
        velocities = np.array(list(self.velocity_history[track_id]))
        speed_variance = np.var([np.linalg.norm(v) for v in velocities])
        
        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + speed_variance)
        return min(confidence, 1.0)
    
    def get_track_statistics(self, track_id: int) -> Dict:
        """Get comprehensive statistics for a track"""
        if track_id not in self.velocity_history or len(self.velocity_history[track_id]) == 0:
            return {}
        
        velocities = np.array(list(self.velocity_history[track_id]))
        speeds = [np.linalg.norm(v) for v in velocities]
        
        return {
            'track_id': track_id,
            'avg_speed_ms': float(np.mean(speeds)),
            'avg_speed_kmh': float(np.mean(speeds) * 3.6),
            'max_speed_ms': float(np.max(speeds)),
            'max_speed_kmh': float(np.max(speeds) * 3.6),
            'speed_variance': float(np.var(speeds)),
            'direction_consistency': self._calculate_direction_consistency(velocities),
            'total_distance': float(np.sum([np.linalg.norm(v) for v in velocities]) * (1/self.fps)),
            'track_duration': len(velocities) / self.fps
        }
    
    def _calculate_direction_consistency(self, velocities: np.ndarray) -> float:
        """Calculate how consistent the direction is (0-1)"""
        if len(velocities) < 2:
            return 0.0
        
        angles = [np.arctan2(v[1], v[0]) for v in velocities]
        
        # Calculate circular variance (direction consistency)
        mean_cos = np.mean([np.cos(a) for a in angles])
        mean_sin = np.mean([np.sin(a) for a in angles])
        
        # Circular variance ranges from 0 (consistent) to 1 (random)
        circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
        
        # Convert to consistency (1 - variance)
        return 1.0 - circular_variance
    
    def visualize_vectors(self, frame: np.ndarray, track_data: Dict) -> np.ndarray:
        """
        Draw velocity vectors on frame
        
        Args:
            frame: Input frame
            track_data: Dictionary with track speed data
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for track_id, data in track_data.items():
            if not data or 'velocity_vector' not in data:
                continue
            
            # Get track positions
            if track_id in self.tracks and len(self.tracks[track_id]) > 0:
                current_pos, _ = self.tracks[track_id][-1]
                center_x, center_y = int(current_pos[0]), int(current_pos[1])
                
                # Get velocity data
                vx, vy = data['velocity_vector']
                speed = data['speed_magnitude']
                
                if speed > 0.1:  # Only draw if moving
                    # Scale vector for visualization
                    scale = 50  # pixels per m/s
                    arrow_length = min(speed * scale, 100)  # Cap arrow length
                    
                    # Calculate arrow end point
                    angle = np.arctan2(vy, vx)
                    end_x = int(center_x + arrow_length * np.cos(angle))
                    end_y = int(center_y + arrow_length * np.sin(angle))
                    
                    # Color based on speed (green=slow, red=fast)
                    speed_ratio = min(speed / 20.0, 1.0)  # Normalize to 20 m/s max
                    color = (0, int(255 * (1 - speed_ratio)), int(255 * speed_ratio))
                    
                    # Draw velocity vector
                    cv2.arrowedLine(annotated_frame, (center_x, center_y), 
                                  (end_x, end_y), color, 2, tipLength=0.3)
                    
                    # Add speed text
                    speed_text = f"{data['speed_kmh']:.1f} km/h"
                    cv2.putText(annotated_frame, speed_text, 
                              (center_x + 10, center_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Add direction text
                    direction_text = f"{data['direction_degrees']:.0f}°"
                    cv2.putText(annotated_frame, direction_text, 
                              (center_x + 10, center_y + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated_frame
    
    def detect_anomalies(self, track_data: Dict) -> List[Dict]:
        """
        Detect speed and direction anomalies
        
        Returns:
            List of anomaly reports
        """
        anomalies = []
        
        for track_id, data in track_data.items():
            if not data or 'speed_magnitude' not in data:
                continue
            
            speed = data['speed_magnitude']
            
            # Speed anomalies
            if speed > 30:  # > 108 km/h (highway speed limit)
                anomalies.append({
                    'track_id': track_id,
                    'type': 'high_speed',
                    'value': speed,
                    'description': f"Vehicle exceeding normal speed: {speed*3.6:.1f} km/h"
                })
            
            elif speed < 0.5 and track_id in self.tracks and len(self.tracks[track_id]) > 5:
                anomalies.append({
                    'track_id': track_id,
                    'type': 'stationary',
                    'value': speed,
                    'description': f"Vehicle appears stationary: {speed*3.6:.1f} km/h"
                })
            
            # Direction change anomalies
            if track_id in self.velocity_history and len(self.velocity_history[track_id]) >= 3:
                recent_directions = []
                for vel in list(self.velocity_history[track_id])[-3:]:
                    if np.linalg.norm(vel) > 0.5:  # Only consider if moving
                        recent_directions.append(np.arctan2(vel[1], vel[0]))
                
                if len(recent_directions) >= 2:
                    angle_changes = []
                    for i in range(1, len(recent_directions)):
                        angle_change = abs(recent_directions[i] - recent_directions[i-1])
                        # Handle angle wrap-around
                        angle_change = min(angle_change, 2*np.pi - angle_change)
                        angle_changes.append(angle_change)
                    
                    max_angle_change = max(angle_changes) if angle_changes else 0
                    
                    if max_angle_change > np.pi/2:  # > 90 degree change
                        anomalies.append({
                            'track_id': track_id,
                            'type': 'sharp_turn',
                            'value': np.degrees(max_angle_change),
                            'description': f"Sharp direction change: {np.degrees(max_angle_change):.1f}°"
                        })
        
        return anomalies