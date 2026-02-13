import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from collections import deque
import time
from picamera2 import Picamera2
import serial
import threading

# ================= CONFIGURATION =================
# Pi Camera Settings
FRAME_WIDTH = 640  # Reduced for Pi performance
FRAME_HEIGHT = 480
CAMERA_FPS = 30  # Realistic for Pi

# Serial Communication (Arduino)
ARDUINO_PORT = '/dev/ttyACM0'  # Change if needed (/dev/ttyUSB0 for some boards)
BAUD_RATE = 9600
ENABLE_SERIAL = True  # Set False for testing without Arduino

# Model Settings
MODEL_NAME = 'buffalo_s'  # Lighter model for Raspberry Pi
CTX_ID = -1  # CPU only on Raspberry Pi (no CUDA)

# Detection Settings - OPTIMIZED FOR RASPBERRY PI
DET_SIZE = (320, 320)  # Reduced for Pi performance
DET_THRESH = 0.50  # Slightly higher for fewer false positives

# Recognition Settings
RECOGNITION_THRESHOLD = 0.38
SMOOTH_FRAMES = 3

# Performance Settings - PI OPTIMIZED
USE_MULTI_SCALE = False  # Disabled for Pi performance
MAX_FACES_TO_PROCESS = 5  # Reduced for Pi
ENABLE_FACE_TRACKING = True

# Detection optimization for Pi
DETECTION_INTERVAL = 5  # More frames between full detection on Pi
PREPROCESS_EVERY_N_FRAMES = 3

# Robot Control Settings
TARGET_CENTER_TOLERANCE = 80  # Pixels from center to consider "aligned"
MOVEMENT_UPDATE_INTERVAL = 0.5  # Seconds between movement commands
STOP_DISTANCE_THRESHOLD = 100  # Will be used with ultrasonic later

# Movement Commands (customize based on your Arduino protocol)
CMD_FORWARD = "F\n"
CMD_BACKWARD = "B\n"
CMD_LEFT = "L\n"
CMD_RIGHT = "R\n"
CMD_STOP = "S\n"
CMD_FORWARD_SLOW = "FS\n"
# =================================================


class ArduinoController:
    """Handles serial communication with Arduino for robot movement"""
    
    def __init__(self, port=ARDUINO_PORT, baud=BAUD_RATE, enabled=ENABLE_SERIAL):
        self.enabled = enabled
        self.serial_conn = None
        self.last_command = None
        self.last_command_time = 0
        
        if self.enabled:
            try:
                self.serial_conn = serial.Serial(port, baud, timeout=1)
                time.sleep(2)  # Wait for Arduino to initialize
                print(f"✅ Arduino connected on {port}")
            except Exception as e:
                print(f"⚠️  Could not connect to Arduino: {e}")
                print("   Running in simulation mode...")
                self.enabled = False
        else:
            print("🔧 Arduino disabled - Simulation Mode")
    
    def send_command(self, command, force=False):
        """Send command to Arduino (debounced to avoid spam)"""
        current_time = time.time()
        
        # Avoid sending same command repeatedly
        if not force and command == self.last_command:
            if (current_time - self.last_command_time) < MOVEMENT_UPDATE_INTERVAL:
                return False
        
        if self.enabled and self.serial_conn:
            try:
                self.serial_conn.write(command.encode())
                self.last_command = command
                self.last_command_time = current_time
                return True
            except Exception as e:
                print(f"⚠️  Serial error: {e}")
                return False
        else:
            # Simulation mode - just print
            cmd_name = {
                CMD_FORWARD: "FORWARD",
                CMD_BACKWARD: "BACKWARD",
                CMD_LEFT: "LEFT",
                CMD_RIGHT: "RIGHT",
                CMD_STOP: "STOP",
                CMD_FORWARD_SLOW: "FORWARD_SLOW"
            }.get(command, command.strip())
            print(f"🤖 Command: {cmd_name}")
            self.last_command = command
            self.last_command_time = current_time
            return True
    
    def stop(self):
        """Emergency stop"""
        return self.send_command(CMD_STOP, force=True)
    
    def close(self):
        """Close serial connection"""
        if self.serial_conn:
            self.stop()
            time.sleep(0.5)
            self.serial_conn.close()
            print("🔌 Arduino connection closed")


class FaceTracker:
    """Lightweight face tracker for inter-frame tracking"""
    
    def __init__(self):
        self.tracker = None
        self.bbox = None
        self.initialized = False
        self.face_data = None
        self.frames_since_update = 0
    
    def init(self, frame, bbox, face_data):
        """Initialize tracker with first detection"""
        self.bbox = bbox
        self.face_data = face_data
        self.tracker = cv2.legacy.TrackerKCF_create()
        
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        self.tracker.init(frame, (x1, y1, w, h))
        self.initialized = True
        self.frames_since_update = 0
    
    def update(self, frame):
        """Update tracker position"""
        if not self.initialized:
            return False, None
        
        success, bbox = self.tracker.update(frame)
        self.frames_since_update += 1
        
        if success:
            x, y, w, h = bbox
            self.bbox = np.array([x, y, x + w, y + h])
            return True, self.bbox
        
        return False, None


class PrizeRobotTracker:
    """Prize Distribution Robot with Face Tracking"""
    
    def __init__(self):
        print("🤖 Initializing Prize Distribution Robot...")
        
        # Initialize Arduino controller
        self.arduino = ArduinoController()
        
        # Initialize FaceAnalysis (CPU mode for Pi)
        print("🧠 Loading AI model (this may take a moment on Pi)...")
        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=['CPUExecutionProvider']  # CPU only for Pi
        )
        self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, det_thresh=DET_THRESH)
        
        # Target storage
        self.target_embeddings = []
        self.target_name = "Unknown"
        
        # Performance tracking
        self.frame_count = 0
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        
        # Face tracking
        self.face_trackers = []
        self.last_full_detection_frame = 0
        
        # Cached preprocessed frame
        self.preprocessed_cache = None
        self.last_preprocessed_frame = -1
        
        # Robot control state
        self.target_locked = False
        self.target_bbox = None
        self.frame_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
        self.robot_state = "IDLE"  # IDLE, SEARCHING, TRACKING, APPROACHING, STOPPED
        
        print("✅ Robot Initialized!")
        print("📹 Pi Camera mode ready")
    
    def load_target(self, image_path):
        """Load target face with multiple embeddings for better accuracy"""
        if not os.path.exists(image_path):
            print("⚠️  Target image not found.")
            return False
        
        print(f"🎯 Loading target from {image_path}...")
        img_target = cv2.imread(image_path)
        
        if img_target is None:
            print("❌ Could not read target image.")
            return False
        
        embeddings_list = []
        
        # Original image
        faces = self.app.get(img_target)
        if len(faces) > 0:
            embeddings_list.append(faces[0].embedding)
        
        # Flipped version
        img_flipped = cv2.flip(img_target, 1)
        faces_flipped = self.app.get(img_flipped)
        if len(faces_flipped) > 0:
            embeddings_list.append(faces_flipped[0].embedding)
        
        # Brightness variations (reduced for Pi)
        for gamma in [0.7, 0.9, 1.1, 1.3]:
            adjusted = self.adjust_gamma(img_target, gamma)
            faces_adj = self.app.get(adjusted)
            if len(faces_adj) > 0:
                embeddings_list.append(faces_adj[0].embedding)
        
        if len(embeddings_list) == 0:
            print("❌ No face found in target image.")
            return False
        
        self.target_embeddings = embeddings_list
        self.target_name = "TARGET"
        print(f"✅ Target locked! Stored {len(embeddings_list)} reference embeddings.")
        return True
    
    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        """Adjust image gamma for brightness variation"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    @staticmethod
    def compute_similarity(feat1, feat2):
        """Compute cosine similarity between two embeddings"""
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    
    def is_target_match(self, embedding):
        """Check if embedding matches target"""
        if not self.target_embeddings:
            return False, 0.0
        
        similarities = np.array([self.compute_similarity(ref_emb, embedding) 
                                for ref_emb in self.target_embeddings])
        max_sim = np.max(similarities)
        
        return max_sim > RECOGNITION_THRESHOLD, max_sim
    
    def preprocess_frame(self, frame):
        """Optimized preprocessing with caching"""
        if self.last_preprocessed_frame == self.frame_count:
            return self.preprocessed_cache
        
        if self.frame_count % PREPROCESS_EVERY_N_FRAMES != 0 and self.preprocessed_cache is not None:
            return self.preprocessed_cache
        
        # Fast histogram equalization
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])
        frame_enhanced = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
        
        self.preprocessed_cache = frame_enhanced
        self.last_preprocessed_frame = self.frame_count
        
        return frame_enhanced
    
    def detect_faces_multiscale(self, frame):
        """Simplified detection for Pi"""
        faces = self.app.get(frame)
        
        if len(faces) > MAX_FACES_TO_PROCESS:
            faces.sort(key=lambda x: x.det_score, reverse=True)
            faces = faces[:MAX_FACES_TO_PROCESS]
        
        return faces
    
    def detect_faces_tracked(self, frame):
        """Hybrid detection: Full detection periodically, tracking in between"""
        
        should_detect = (
            (self.frame_count - self.last_full_detection_frame) >= DETECTION_INTERVAL or
            not ENABLE_FACE_TRACKING or
            len(self.face_trackers) == 0
        )
        
        if should_detect:
            faces = self.detect_faces_multiscale(frame)
            self.last_full_detection_frame = self.frame_count
            
            if ENABLE_FACE_TRACKING:
                self.face_trackers = []
                for face in faces:
                    tracker = FaceTracker()
                    tracker.init(frame, face.bbox, face)
                    self.face_trackers.append(tracker)
            
            return faces
        
        # Use tracking for intermediate frames
        tracked_faces = []
        valid_trackers = []
        
        for tracker in self.face_trackers:
            success, bbox = tracker.update(frame)
            
            if success and tracker.frames_since_update < DETECTION_INTERVAL:
                tracker.face_data.bbox = bbox
                tracked_faces.append(tracker.face_data)
                valid_trackers.append(tracker)
        
        self.face_trackers = valid_trackers
        return tracked_faces
    
    def calculate_movement_command(self, target_bbox):
        """Calculate robot movement based on target position"""
        if target_bbox is None:
            return CMD_STOP, "NO_TARGET"
        
        # Calculate target center
        x1, y1, x2, y2 = target_bbox.astype(int)
        target_center_x = (x1 + x2) // 2
        target_center_y = (y1 + y2) // 2
        
        # Calculate offset from frame center
        offset_x = target_center_x - self.frame_center[0]
        offset_y = target_center_y - self.frame_center[1]
        
        # Calculate target size (for distance estimation)
        target_width = x2 - x1
        target_height = y2 - y1
        target_size = (target_width + target_height) / 2
        
        # Decision logic
        # Horizontal alignment first
        if abs(offset_x) > TARGET_CENTER_TOLERANCE:
            if offset_x > 0:
                return CMD_RIGHT, "ALIGN_RIGHT"
            else:
                return CMD_LEFT, "ALIGN_LEFT"
        
        # Check if close enough (using size as proxy for distance)
        # Larger face = closer to robot
        if target_size > 200:  # Adjust based on testing
            return CMD_STOP, "TARGET_REACHED"
        
        # Move forward if aligned but not close
        return CMD_FORWARD_SLOW, "APPROACHING"
    
    def control_robot(self, faces):
        """Main robot control logic"""
        target_found = False
        target_bbox = None
        
        # Find target in detected faces
        for face in faces:
            if self.target_embeddings:
                is_match, similarity = self.is_target_match(face.embedding)
                if is_match:
                    target_found = True
                    target_bbox = face.bbox
                    break
        
        # Update robot state and send commands
        if target_found:
            self.target_locked = True
            self.target_bbox = target_bbox
            
            # Calculate and send movement command
            command, state = self.calculate_movement_command(target_bbox)
            self.robot_state = state
            self.arduino.send_command(command)
            
        else:
            if self.target_locked:
                # Just lost target - stop
                self.arduino.send_command(CMD_STOP)
                self.robot_state = "TARGET_LOST"
            else:
                # Never had target - searching
                self.robot_state = "SEARCHING"
                # Optional: slow rotation to search
                # self.arduino.send_command(CMD_LEFT)
            
            self.target_locked = False
            self.target_bbox = None
    
    def draw_results(self, frame, faces):
        """Draw detection results and robot status"""
        target_found = False
        
        for face in faces:
            box = face.bbox.astype(int)
            
            color = (0, 0, 255)
            label = f"{face.det_score:.2f}"
            thickness = 2
            
            if self.target_embeddings:
                is_match, similarity = self.is_target_match(face.embedding)
                
                if is_match:
                    color = (0, 255, 0)
                    label = f"TARGET {similarity:.2f}"
                    thickness = 3
                    target_found = True
                    
                    # Draw crosshair on target
                    center_x = (box[0] + box[2]) // 2
                    center_y = (box[1] + box[3]) // 2
                    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
                    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
                else:
                    continue  # Skip non-targets
            
            # Draw bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)
            
            # Label
            cv2.putText(frame, label, (box[0], box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw frame center crosshair
        cv2.line(frame, (self.frame_center[0] - 30, self.frame_center[1]), 
                (self.frame_center[0] + 30, self.frame_center[1]), (255, 0, 0), 1)
        cv2.line(frame, (self.frame_center[0], self.frame_center[1] - 30), 
                (self.frame_center[0], self.frame_center[1] + 30), (255, 0, 0), 1)
        
        # Status display
        status_color = (0, 255, 0) if target_found else (0, 0, 255)
        
        cv2.rectangle(frame, (10, 10), (220, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (220, 100), status_color, 2)
        
        cv2.putText(frame, f"State: {self.robot_state}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Target: {'LOCKED' if target_found else 'SEARCHING'}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        cv2.putText(frame, f"Faces: {len(faces)}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def update_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)
    
    def draw_ui(self, frame):
        """Draw UI overlay"""
        fps = self.update_fps()
        
        # FPS display
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 2)
        
        return frame
    
    def run(self):
        """Main loop - Pi Camera optimized"""
        # Load target
        image_name = input("📂 Target image name (or Enter to skip): ").strip()
        
        if image_name:
            if not self.load_target(image_name):
                print("❌ Failed to load target. Exiting.")
                return
        else:
            print("⚠️  No target loaded. Robot will not move.")
            print("   Load a target to enable autonomous navigation.")
        
        # Initialize Pi Camera
        print("📹 Initializing Pi Camera...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": CAMERA_FPS}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Camera warm-up
        
        print("🎥 Camera Started!")
        print("🤖 Robot Control Active!")
        print("Controls: q=quit | s=save | r=reload | SPACE=emergency stop")
        
        try:
            while True:
                # Capture frame from Pi Camera
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
                
                self.frame_count += 1
                
                # Preprocess
                frame_processed = self.preprocess_frame(frame)
                
                # Detect faces
                faces = self.detect_faces_tracked(frame_processed)
                
                # Control robot based on detections
                if self.target_embeddings:
                    self.control_robot(faces)
                
                # Draw results
                frame_display = self.draw_results(frame.copy(), faces)
                frame_display = self.draw_ui(frame_display)
                
                # Display
                cv2.imshow('Prize Robot', frame_display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quit command received")
                    break
                elif key == ord(' '):
                    print("⚠️  EMERGENCY STOP")
                    self.arduino.stop()
                    self.robot_state = "EMERGENCY_STOP"
                elif key == ord('s'):
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved: {filename}")
                elif key == ord('r') and image_name:
                    print("🔄 Reloading target...")
                    self.arduino.stop()
                    self.load_target(image_name)
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            print("🧹 Cleaning up...")
            self.arduino.stop()
            time.sleep(0.5)
            picam2.stop()
            cv2.destroyAllWindows()
            self.arduino.close()
            print(f"👋 Stopped. Average FPS: {np.mean(self.fps_buffer):.1f}")


if __name__ == "__main__":
    robot = PrizeRobotTracker()
    robot.run()
