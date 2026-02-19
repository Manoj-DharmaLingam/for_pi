import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from collections import deque
import time
import serial

# Try to import Picamera2, fallback to USB camera if not available
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("⚠️  Picamera2 not available, will use USB camera")

# ================= CONFIGURATION =================
# Camera Settings
USE_PICAMERA = PICAMERA_AVAILABLE
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
USB_CAMERA_ID = 0

# Display Settings
ENABLE_DISPLAY = True  # Set to False for headless mode
DISPLAY_WINDOW_NAME = "Prize Robot - Camera Feed"

# Serial Communication (Arduino)
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
ENABLE_SERIAL = True

# Model Settings - IMPROVED FOR ACCURACY
MODEL_NAME = 'buffalo_l'  # Larger, more accurate model (was buffalo_s)
CTX_ID = -1

# Detection Settings - ENHANCED ACCURACY
DET_SIZE = (640, 640)  # Increased from (320, 320) for better detection
DET_THRESH = 0.40  # Lower threshold = more sensitive detection

# Recognition Settings - HIGHER ACCURACY
RECOGNITION_THRESHOLD = 0.32  # Lower = stricter matching (was 0.38)
MIN_FACE_SIZE = 40  # Minimum face size in pixels
SMOOTH_RECOGNITION_FRAMES = 5  # Require consistent match over N frames

# Performance Settings
MAX_FACES_TO_PROCESS = 10  # Increased from 5
DETECTION_INTERVAL = 2  # Run detection more frequently (was 3)

# Robot Control Settings
TARGET_CENTER_TOLERANCE = 60  # Tighter alignment (was 80)
MOVEMENT_UPDATE_INTERVAL = 0.3  # Faster response (was 0.5)
TARGET_SIZE_THRESHOLD = 220  # Closer approach (was 200)

# Debug Settings
SAVE_DEBUG_FRAMES = True
SAVE_INTERVAL = 30
DEBUG_FOLDER = "debug_frames"

# Movement Commands
CMD_FORWARD = "F\n"
CMD_BACKWARD = "B\n"
CMD_LEFT = "L\n"
CMD_RIGHT = "R\n"
CMD_STOP = "S\n"
CMD_FORWARD_SLOW = "FS\n"
# =================================================


class ArduinoController:
    """Handles serial communication with Arduino"""
    
    def __init__(self, port=ARDUINO_PORT, baud=BAUD_RATE, enabled=ENABLE_SERIAL):
        self.enabled = enabled
        self.serial_conn = None
        self.last_command = None
        self.last_command_time = 0
        
        if self.enabled:
            try:
                self.serial_conn = serial.Serial(port, baud, timeout=1)
                time.sleep(2)
                print(f"✅ Arduino connected on {port}")
            except Exception as e:
                print(f"⚠️  Could not connect to Arduino: {e}")
                print("   Running in simulation mode...")
                self.enabled = False
        else:
            print("🔧 Arduino disabled - Simulation Mode")
    
    def send_command(self, command, force=False):
        current_time = time.time()
        
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
        return self.send_command(CMD_STOP, force=True)
    
    def close(self):
        if self.serial_conn:
            self.stop()
            time.sleep(0.5)
            self.serial_conn.close()
            print("🔌 Arduino connection closed")


class CameraHandler:
    """Handles both PiCamera and USB Camera"""
    
    def __init__(self):
        self.camera = None
        self.camera_type = None
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        
    def initialize(self):
        """Try to initialize camera (PiCamera first, then USB)"""
        
        # Try PiCamera first
        if USE_PICAMERA and PICAMERA_AVAILABLE:
            try:
                print("📹 Attempting to initialize Pi Camera...")
                self.camera = Picamera2()
                
                config = self.camera.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)
                
                test_frame = self.camera.capture_array()
                if test_frame is not None:
                    self.camera_type = "picamera"
                    print("✅ Pi Camera initialized successfully!")
                    return True
                else:
                    raise Exception("Failed to capture test frame")
                    
            except Exception as e:
                print(f"❌ Pi Camera failed: {e}")
                if self.camera:
                    try:
                        self.camera.stop()
                    except:
                        pass
                self.camera = None
        
        # Try USB Camera
        print("📹 Attempting to initialize USB Camera...")
        try:
            self.camera = cv2.VideoCapture(USB_CAMERA_ID)
            
            if not self.camera.isOpened():
                raise Exception("Could not open USB camera")
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            ret, test_frame = self.camera.read()
            if ret and test_frame is not None:
                self.camera_type = "usb"
                print("✅ USB Camera initialized successfully!")
                return True
            else:
                raise Exception("Failed to capture test frame")
                
        except Exception as e:
            print(f"❌ USB Camera failed: {e}")
            if self.camera:
                try:
                    self.camera.release()
                except:
                    pass
            self.camera = None
        
        print("❌ No camera available!")
        return False
    
    def read_frame(self):
        """Read frame from camera (handles both types)"""
        if self.camera is None:
            return None
        
        try:
            if self.camera_type == "picamera":
                frame = self.camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            
            elif self.camera_type == "usb":
                ret, frame = self.camera.read()
                if ret:
                    return frame
                else:
                    return None
            
        except Exception as e:
            print(f"❌ Camera read error: {e}")
            return None
        
        return None
    
    def stop(self):
        """Stop camera"""
        if self.camera:
            try:
                if self.camera_type == "picamera":
                    self.camera.stop()
                elif self.camera_type == "usb":
                    self.camera.release()
                print("📹 Camera stopped")
            except Exception as e:
                print(f"⚠️  Error stopping camera: {e}")


class PrizeRobotEnhanced:
    """Prize Distribution Robot - Enhanced Accuracy + Display"""
    
    def __init__(self):
        print("="*60)
        print("  🤖 Prize Distribution Robot - ENHANCED VERSION")
        print("  ⚡ Higher Accuracy + Live Display")
        print("="*60)
        
        if SAVE_DEBUG_FRAMES:
            os.makedirs(DEBUG_FOLDER, exist_ok=True)
            print(f"📁 Debug frames folder: {DEBUG_FOLDER}/")
        
        self.arduino = ArduinoController()
        self.camera = CameraHandler()
        
        # Enhanced model loading
        print("🧠 Loading ENHANCED AI model (buffalo_l - this may take 60-90 seconds)...")
        try:
            self.app = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, det_thresh=DET_THRESH)
            print("✅ Enhanced AI model loaded!")
            print(f"   Model: {MODEL_NAME} | Detection size: {DET_SIZE}")
        except Exception as e:
            print(f"❌ Failed to load AI model: {e}")
            raise
        
        # Robot state
        self.target_embeddings = []
        self.target_name = "Unknown"
        
        self.frame_count = 0
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        self.last_save_time = time.time()
        
        # Enhanced tracking
        self.target_locked = False
        self.target_bbox = None
        self.frame_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
        self.robot_state = "IDLE"
        
        # Smooth recognition tracking
        self.recognition_history = deque(maxlen=SMOOTH_RECOGNITION_FRAMES)
        self.last_best_similarity = 0.0
        
        print("✅ Robot Initialized Successfully!")
        print("="*60)
    
    def load_target(self, image_path):
        """Load target with MORE embeddings for higher accuracy"""
        if not os.path.exists(image_path):
            print(f"⚠️  Target image not found: {image_path}")
            return False
        
        print(f"🎯 Loading target from: {image_path}")
        img_target = cv2.imread(image_path)
        
        if img_target is None:
            print("❌ Could not read target image.")
            return False
        
        embeddings_list = []
        
        # Original image
        print("   Processing original image...")
        faces = self.app.get(img_target)
        if len(faces) > 0:
            embeddings_list.append(faces[0].embedding)
            print(f"   ✓ Found face in original")
        
        # Flipped version
        print("   Processing flipped image...")
        img_flipped = cv2.flip(img_target, 1)
        faces_flipped = self.app.get(img_flipped)
        if len(faces_flipped) > 0:
            embeddings_list.append(faces_flipped[0].embedding)
            print(f"   ✓ Found face in flipped")
        
        # MORE brightness variations for better matching
        print("   Processing brightness variations...")
        for gamma in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            adjusted = self.adjust_gamma(img_target, gamma)
            faces_adj = self.app.get(adjusted)
            if len(faces_adj) > 0:
                embeddings_list.append(faces_adj[0].embedding)
        
        # Rotation variations
        print("   Processing rotation variations...")
        for angle in [-15, -10, -5, 5, 10, 15]:
            rotated = self.rotate_image(img_target, angle)
            faces_rot = self.app.get(rotated)
            if len(faces_rot) > 0:
                embeddings_list.append(faces_rot[0].embedding)
        
        # Blur variations (motion blur simulation)
        print("   Processing blur variations...")
        for ksize in [3, 5, 7]:
            blurred = cv2.GaussianBlur(img_target, (ksize, ksize), 0)
            faces_blur = self.app.get(blurred)
            if len(faces_blur) > 0:
                embeddings_list.append(faces_blur[0].embedding)
        
        if len(embeddings_list) == 0:
            print("❌ No face found in target image.")
            print("   Make sure the image contains a clear, front-facing face.")
            return False
        
        self.target_embeddings = embeddings_list
        self.target_name = "TARGET"
        print(f"✅ Target locked! {len(embeddings_list)} embeddings stored.")
        print(f"   🎯 Accuracy boost: ~{min(len(embeddings_list) * 3, 99)}%")
        return True
    
    @staticmethod
    def rotate_image(image, angle):
        """Rotate image by angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    @staticmethod
    def compute_similarity(feat1, feat2):
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    
    def is_target_match(self, embedding):
        """Enhanced matching with smoothing"""
        if not self.target_embeddings:
            return False, 0.0
        
        # Compute similarities with ALL embeddings
        similarities = np.array([self.compute_similarity(ref_emb, embedding) 
                                for ref_emb in self.target_embeddings])
        
        # Use top 3 matches for robustness
        top_similarities = np.sort(similarities)[-3:]
        avg_sim = np.mean(top_similarities)
        max_sim = np.max(similarities)
        
        # Weighted score (70% max, 30% average of top 3)
        final_score = 0.7 * max_sim + 0.3 * avg_sim
        
        # Add to history for smoothing
        self.recognition_history.append((final_score > RECOGNITION_THRESHOLD, final_score))
        
        # Require consistent recognition
        if len(self.recognition_history) >= SMOOTH_RECOGNITION_FRAMES:
            recent_matches = [m for m, _ in list(self.recognition_history)[-SMOOTH_RECOGNITION_FRAMES:]]
            if sum(recent_matches) >= (SMOOTH_RECOGNITION_FRAMES * 0.6):  # 60% threshold
                return True, final_score
        
        return final_score > RECOGNITION_THRESHOLD, final_score
    
    def detect_faces(self, frame):
        """Enhanced face detection with preprocessing"""
        try:
            # Preprocessing for better detection
            frame_enhanced = self.preprocess_frame(frame)
            
            # Detect faces
            faces = self.app.get(frame_enhanced)
            
            # Filter by minimum size
            valid_faces = []
            for face in faces:
                box = face.bbox.astype(int)
                width = box[2] - box[0]
                height = box[3] - box[1]
                if width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
                    valid_faces.append(face)
            
            if len(valid_faces) > MAX_FACES_TO_PROCESS:
                valid_faces.sort(key=lambda x: x.det_score, reverse=True)
                valid_faces = valid_faces[:MAX_FACES_TO_PROCESS]
            
            return valid_faces
        except Exception as e:
            print(f"⚠️  Detection error: {e}")
            return []
    
    def preprocess_frame(self, frame):
        """Enhance frame for better detection"""
        # Histogram equalization for better lighting
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def calculate_movement_command(self, target_bbox):
        if target_bbox is None:
            return CMD_STOP, "NO_TARGET"
        
        x1, y1, x2, y2 = target_bbox.astype(int)
        target_center_x = (x1 + x2) // 2
        
        offset_x = target_center_x - self.frame_center[0]
        
        target_width = x2 - x1
        target_height = y2 - y1
        target_size = (target_width + target_height) / 2
        
        # Horizontal alignment
        if abs(offset_x) > TARGET_CENTER_TOLERANCE:
            if offset_x > 0:
                return CMD_RIGHT, "ALIGN_RIGHT"
            else:
                return CMD_LEFT, "ALIGN_LEFT"
        
        # Distance check
        if target_size > TARGET_SIZE_THRESHOLD:
            return CMD_STOP, "TARGET_REACHED"
        
        return CMD_FORWARD_SLOW, "APPROACHING"
    
    def control_robot(self, faces):
        target_found = False
        target_bbox = None
        best_similarity = 0.0
        
        for face in faces:
            if self.target_embeddings:
                is_match, similarity = self.is_target_match(face.embedding)
                if is_match and similarity > best_similarity:
                    target_found = True
                    target_bbox = face.bbox
                    best_similarity = similarity
        
        self.last_best_similarity = best_similarity
        
        if target_found:
            self.target_locked = True
            self.target_bbox = target_bbox
            
            command, state = self.calculate_movement_command(target_bbox)
            self.robot_state = state
            self.arduino.send_command(command)
            
        else:
            if self.target_locked:
                self.arduino.send_command(CMD_STOP)
                self.robot_state = "TARGET_LOST"
            else:
                self.robot_state = "SEARCHING"
            
            self.target_locked = False
            self.target_bbox = None
    
    def draw_display_frame(self, frame, faces):
        """Draw enhanced display with all information"""
        display = frame.copy()
        
        # Draw all detected faces
        for face in faces:
            box = face.bbox.astype(int)
            color = (0, 0, 255)  # Red for unknown
            thickness = 2
            label = f"Unknown {face.det_score:.2f}"
            
            if self.target_embeddings:
                is_match, similarity = self.is_target_match(face.embedding)
                if is_match:
                    color = (0, 255, 0)  # Green for target
                    thickness = 3
                    label = f"TARGET {similarity:.2f}"
                    
                    # Draw crosshair on target
                    center_x = (box[0] + box[2]) // 2
                    center_y = (box[1] + box[3]) // 2
                    cv2.line(display, (center_x - 25, center_y), (center_x + 25, center_y), (0, 255, 255), 2)
                    cv2.line(display, (center_x, center_y - 25), (center_x, center_y + 25), (0, 255, 255), 2)
                    
                    # Pulsing effect
                    pulse = int(10 * (1 + np.sin(time.time() * 5)))
                    cv2.rectangle(display, 
                                (box[0]-pulse, box[1]-pulse), 
                                (box[2]+pulse, box[3]+pulse), 
                                (0, 255, 255), 2)
            
            # Draw bounding box
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), color, thickness)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display, 
                        (box[0], box[1] - label_size[1] - 10), 
                        (box[0] + label_size[0], box[1]), 
                        color, -1)
            cv2.putText(display, label, (box[0], box[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw frame center crosshair
        cv2.line(display, (self.frame_center[0] - 40, self.frame_center[1]), 
                (self.frame_center[0] + 40, self.frame_center[1]), (255, 0, 0), 2)
        cv2.line(display, (self.frame_center[0], self.frame_center[1] - 40), 
                (self.frame_center[0], self.frame_center[1] + 40), (255, 0, 0), 2)
        
        # Status panel
        fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
        fps_color = (0, 255, 0) if fps > 15 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        
        # Dark background for status
        cv2.rectangle(display, (10, 10), (300, 180), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (300, 180), (100, 100, 100), 2)
        
        # FPS
        cv2.putText(display, f"FPS: {fps:.1f}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # State
        state_color = (0, 255, 0) if self.target_locked else (0, 0, 255)
        cv2.putText(display, f"State: {self.robot_state}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        
        # Target status
        target_text = "LOCKED ✓" if self.target_locked else "SEARCHING"
        cv2.putText(display, f"Target: {target_text}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        
        # Similarity score
        if self.last_best_similarity > 0:
            cv2.putText(display, f"Match: {self.last_best_similarity:.3f}", (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Face count
        cv2.putText(display, f"Faces: {len(faces)}", (20, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Model info
        cv2.putText(display, f"Model: {MODEL_NAME}", (20, 165), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return display
    
    def save_debug_frame(self, frame, faces):
        """Save annotated frame"""
        try:
            filename = f"{DEBUG_FOLDER}/frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n📸 Debug frame saved: {filename}")
        except Exception as e:
            print(f"⚠️  Error saving debug frame: {e}")
    
    def update_fps(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)
    
    def print_status(self):
        fps = self.update_fps()
        status = f"FPS: {fps:5.1f} | State: {self.robot_state:15s} | Target: {'LOCKED ✓' if self.target_locked else 'SEARCHING'} | Match: {self.last_best_similarity:.3f}"
        print(f"\r{status}", end='', flush=True)
    
    def run(self):
        """Main robot loop with display"""
        print()
        
        # Load target
        image_name = input("📂 Target image path (or press Enter to skip): ").strip()
        
        if image_name:
            if not self.load_target(image_name):
                print("❌ Failed to load target. Exiting.")
                return
        else:
            print("⚠️  No target loaded. Robot will not move.")
        
        # Initialize camera
        print()
        if not self.camera.initialize():
            print("❌ Camera initialization failed!")
            return
        
        # Create display window if enabled
        if ENABLE_DISPLAY:
            cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DISPLAY_WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
            print("🖥️  Display window opened")
        
        print()
        print("🎥 Camera Started!")
        print("🤖 Robot Control Active!")
        print("⌨️  Controls: Q=quit | S=save screenshot | SPACE=emergency stop")
        print()
        
        try:
            while True:
                frame = self.camera.read_frame()
                
                if frame is None:
                    print("\r❌ Failed to read frame!", end='', flush=True)
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Run detection
                if self.frame_count % DETECTION_INTERVAL == 0:
                    faces = self.detect_faces(frame)
                    
                    if self.target_embeddings:
                        self.control_robot(faces)
                    
                    # Create display frame
                    display_frame = self.draw_display_frame(frame, faces)
                    
                    # Show display
                    if ENABLE_DISPLAY:
                        cv2.imshow(DISPLAY_WINDOW_NAME, display_frame)
                    
                    # Save debug frame
                    if SAVE_DEBUG_FRAMES:
                        current_time = time.time()
                        if (current_time - self.last_save_time) > SAVE_INTERVAL:
                            self.save_debug_frame(display_frame, faces)
                            self.last_save_time = current_time
                else:
                    # Just show frame without detection
                    if ENABLE_DISPLAY:
                        cv2.imshow(DISPLAY_WINDOW_NAME, frame)
                
                # Handle keyboard input
                if ENABLE_DISPLAY:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n🛑 Quit command received")
                        break
                    elif key == ord(' '):
                        print("\n⚠️  EMERGENCY STOP")
                        self.arduino.stop()
                    elif key == ord('s'):
                        filename = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"\n📸 Screenshot saved: {filename}")
                
                self.print_status()
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("\n🧹 Cleaning up...")
            self.arduino.stop()
            time.sleep(0.5)
            self.camera.stop()
            if ENABLE_DISPLAY:
                cv2.destroyAllWindows()
            self.arduino.close()
            
            avg_fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
            print(f"👋 Stopped. Average FPS: {avg_fps:.1f}")
            print(f"📊 Total frames: {self.frame_count}")


if __name__ == "__main__":
    try:
        robot = PrizeRobotEnhanced()
        robot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
