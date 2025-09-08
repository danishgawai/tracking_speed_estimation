# Vector-Based Speed Estimation Configuration Guide

## Quick Start

1. **Place the files in your project structure:**
   ```
   your_project/
   ├── main.py          # Enhanced main script
   ├── utils/
   │   ├── vector_speed_estimator.py
   │   ├── yolov8_infer.py
   │   └── ...
   └── tracker/
       ├── byte_tracker.py
       └── ...
   ```

2. **Install additional dependencies:**
   ```bash
   pip install opencv-python numpy scipy
   ```

3. **Run the enhanced system:**
   ```bash
   python main.py
   ```

## Drone Motion Compensation Methods

### 1. Homography-based Compensation (Recommended)
- **Best for:** Stable drone flights with consistent altitude
- **Accuracy:** High (handles perspective changes)
- **Computational cost:** Medium

```python
speed_estimator = VectorSpeedEstimator(
    motion_compensation_method="homography"
)
```

### 2. Optical Flow-based Compensation
- **Best for:** Rapid drone movements, low altitude flights
- **Accuracy:** Good (faster but less precise)
- **Computational cost:** Low

```python
speed_estimator = VectorSpeedEstimator(
    motion_compensation_method="optical_flow"
)
```

### 3. Hybrid Compensation
- **Best for:** Complex flight patterns with varying conditions
- **Accuracy:** Very High (combines both methods)
- **Computational cost:** High

```python
speed_estimator = VectorSpeedEstimator(
    motion_compensation_method="hybrid"
)
```

## Calibration Parameters

### Ground Sampling Distance (GSD) Calibration
Critical for accurate real-world measurements:

```python
# Method 1: Using drone telemetry
speed_estimator.update_gsd(
    altitude_m=100,           # Drone altitude in meters
    focal_length_mm=4.5,      # Camera focal length
    sensor_width_mm=6.17,     # Camera sensor width
    image_width_px=1920       # Image width in pixels
)

# Method 2: Using known ground references
# Measure real-world distance between two points in the image
# Calculate GSD = real_distance_meters / pixel_distance
speed_estimator.gsd = 0.15  # Example: 15 cm per pixel
```

### Performance Tuning Parameters

```python
speed_estimator = VectorSpeedEstimator(
    fps=30,                          # Video frame rate
    homography_update_interval=5,     # Update every 5 frames (balance accuracy/speed)
    smoothing_window=5,              # Velocity smoothing window
    motion_compensation_method="homography"
)
```

## Key Features

### 1. Vector-Based Speed Calculation
- Provides both speed magnitude and direction
- Handles vehicles moving in any direction
- Compensates for drone motion automatically

### 2. Advanced Motion Compensation
- **Feature Matching:** Uses ORB features for robust motion estimation
- **Homography Transformation:** Handles perspective changes
- **Kalman Filtering:** Smooths velocity estimates

### 3. Comprehensive Analytics
- Speed statistics per track
- Direction consistency analysis
- Anomaly detection (speeding, stationary vehicles, sharp turns)

### 4. Real-time Visualization
- Velocity vectors overlaid on video
- Color-coded speed indicators
- Live statistics display

## Output Files

The system generates several output files:

1. **Enhanced Video:** `video_out_vector_speed_[timestamp].mp4`
   - Contains velocity vectors, speed annotations, track trajectories

2. **Speed Data:** `speed_data_[timestamp].json`
   - Frame-by-frame speed data for all tracks
   - JSON format for easy analysis

3. **Final Statistics:** `final_statistics_[timestamp].json`
   - Comprehensive statistics for each track
   - Average/max speeds, direction consistency, total distance

## Advanced Configuration

### Anomaly Detection Thresholds

```python
# In vector_speed_estimator.py, modify detect_anomalies() method:

def detect_anomalies(self, track_data: Dict) -> List[Dict]:
    anomalies = []
    
    for track_id, data in track_data.items():
        speed = data['speed_magnitude']
        
        # Customize thresholds based on your scenario
        if speed > 25:  # Adjust for highway (25 m/s = 90 km/h)
            anomalies.append({
                'type': 'high_speed',
                'value': speed,
                'description': f"High speed detected: {speed*3.6:.1f} km/h"
            })
```

### Motion Compensation Fine-tuning

```python
# Adjust feature detection parameters in _detect_and_match_features():
self.orb = cv2.ORB_create(
    nfeatures=1000,      # Increase for better motion estimation
    scaleFactor=1.2,     # Multi-scale detection
    nlevels=8           # Pyramid levels
)

# Adjust homography estimation parameters:
H, mask = cv2.findHomography(
    src_pts, dst_pts, 
    cv2.RANSAC, 
    5.0,                # Reduce for stricter matching
    maxIters=2000       # Increase for better estimation
)
```

## Best Practices for Drone Motion Compensation

### 1. Feature Matching Approach
- **Pros:** Most accurate, handles complex transformations
- **Cons:** Computationally intensive
- **Use when:** High accuracy required, stable processing environment

### 2. Optical Flow Approach  
- **Pros:** Fast, robust to lighting changes
- **Cons:** Less accurate for large motions
- **Use when:** Real-time processing critical, rapid drone movements

### 3. Hybrid Approach
- **Pros:** Best of both worlds, adaptive
- **Cons:** Most computationally expensive
- **Use when:** Maximum accuracy needed, computational resources available

## Troubleshooting

### Poor Speed Accuracy
1. **Check GSD calibration** - Most common issue
2. **Verify drone altitude data** - Critical for GSD calculation
3. **Adjust motion compensation method** - Try different methods

### Motion Compensation Failure
1. **Insufficient features** - Increase ORB feature count
2. **Poor image quality** - Improve camera settings
3. **Rapid drone movement** - Switch to optical flow method

### Performance Issues
1. **Reduce homography update interval** - Update every 10 frames instead of 5
2. **Decrease smoothing window** - Use fewer frames for averaging
3. **Switch to optical flow** - Faster motion compensation

## Integration with Existing Systems

The vector speed estimator is designed to integrate seamlessly with:
- **YOLO detectors** (any version)
- **ByteTracker** (and other multi-object trackers)
- **OpenCV-based pipelines**

Simply replace scalar speed calculations with vector-based methods while maintaining your existing detection and tracking infrastructure.
