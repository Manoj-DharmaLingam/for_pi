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
USE_PICAMERA = PICAMERA_AVAILABLE  # Set to False to force USB camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
USB_CAMERA_ID = 0  # Change if you have multiple cameras

# Serial Communication (Arduino)
ARDUINO_PORT = '/dev/ttyACM0'  # Change if needed (/dev/ttyUSB0)
BAUD_RATE = 9600
ENABLE_SERIAL = True  # Set False for testing without Arduino

# Model Settings
MODEL_NAME = 'buffalo_s'
CTX_ID = -1  # CPU only on Raspberry Pi

# Detection Settings
DET_SIZE = (320, 320)
DET_THRESH = 0.50

# Recognition Settings
RECOGNITION_THRESHOLD = 0.38

# Performance Settings
MAX_FACES_TO_PROCESS = 5
DETECTION_INTERVAL = 3  # Run detection every N frames

# Robot Control Settings
TARGET_CENTER_TOLERANCE = 80
MOVEMENT_UPDATE_INTERVAL = 0.5

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
                
                # Configure camera
                config = self.camera.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)  # Camera warm-up
                
                # Test capture
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
            
            # Test capture
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
                # Convert RGB to BGR for OpenCV
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


class PrizeRobot:
    """Prize Distribution Robot - Complete Working Version"""
    
    def __init__(self):
        print("="*60)
        print("  🤖 Prize Distribution Robot - Initializing")
        print("="*60)
        
        # Create debug folder
        if SAVE_DEBUG_FRAMES:
            os.makedirs(DEBUG_FOLDER, exist_ok=True)
            print(f"📁 Debug frames folder: {DEBUG_FOLDER}/")
        
        # Initialize Arduino
        self.arduino = ArduinoController()
        
        # Initialize camera
        self.camera = CameraHandler()
        
        # Initialize face detection
        print("🧠 Loading AI model (this may take 30-60 seconds)...")
        try:
            self.app = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, det_thresh=DET_THRESH)
            print("✅ AI model loaded!")
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
        
        self.target_locked = False
        self.target_bbox = None
        self.frame_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
        self.robot_state = "IDLE"
        
        print("✅ Robot Initialized Successfully!")
        print("="*60)
    
    def load_target(self, image_path):
        """Load target face with multiple embeddings"""
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
        
        # Brightness variations
        print("   Processing brightness variations...")
        for gamma in [0.7, 0.9, 1.1, 1.3]:
            adjusted = self.adjust_gamma(img_target, gamma)
            faces_adj = self.app.get(adjusted)
            if len(faces_adj) > 0:
                embeddings_list.append(faces_adj[0].embedding)
        
        if len(embeddings_list) == 0:
            print("❌ No face found in target image.")
            print("   Make sure the image contains a clear, front-facing face.")
            return False
        
        self.target_embeddings = embeddings_list
        self.target_name = "TARGET"
        print(f"✅ Target locked! {len(embeddings_list)} embeddings stored.")
        return True
    
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
        if not self.target_embeddings:
            return False, 0.0
        
        similarities = np.array([self.compute_similarity(ref_emb, embedding) 
                                for ref_emb in self.target_embeddings])
        max_sim = np.max(similarities)
        
        return max_sim > RECOGNITION_THRESHOLD, max_sim
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        try:
            faces = self.app.get(frame)
            
            if len(faces) > MAX_FACES_TO_PROCESS:
                faces.sort(key=lambda x: x.det_score, reverse=True)
                faces = faces[:MAX_FACES_TO_PROCESS]
            
            return faces
        except Exception as e:
            print(f"⚠️  Detection error: {e}")
            return []
    
    def calculate_movement_command(self, target_bbox):
        if target_bbox is None:
            return CMD_STOP, "NO_TARGET"
        
        x1, y1, x2, y2 = target_bbox.astype(int)
        target_center_x = (x1 + x2) // 2
        
        offset_x = target_center_x - self.frame_center[0]
        
        target_width = x2 - x1
        target_height = y2 - y1
        target_size = (target_width + target_height) / 2
        
        # Horizontal alignment first
        if abs(offset_x) > TARGET_CENTER_TOLERANCE:
            if offset_x > 0:
                return CMD_RIGHT, "ALIGN_RIGHT"
            else:
                return CMD_LEFT, "ALIGN_LEFT"
        
        # Check if close enough (adjust threshold based on testing)
        if target_size > 200:
            return CMD_STOP, "TARGET_REACHED"
        
        # Move forward if aligned
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
    
    def save_debug_frame(self, frame, faces):
        """Save annotated frame for debugging"""
        try:
            frame_debug = frame.copy()
            
            # Draw detections
            for face in faces:
                box = face.bbox.astype(int)
                color = (0, 0, 255)
                
                if self.target_embeddings:
                    is_match, similarity = self.is_target_match(face.embedding)
                    if is_match:
                        color = (0, 255, 0)
                        label = f"TARGET {similarity:.2f}"
                    else:
                        label = f"OTHER {face.det_score:.2f}"
                else:
                    label = f"{face.det_score:.2f}"
                
                cv2.rectangle(frame_debug, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame_debug, label, (box[0], box[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw status
            fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
            cv2.putText(frame_debug, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_debug, f"State: {self.robot_state}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_debug, f"Locked: {self.target_locked}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame_debug, f"Faces: {len(faces)}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save
            filename = f"{DEBUG_FOLDER}/frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame_debug)
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
        faces_indicator = "●" if self.frame_count % 6 < 3 else "○"
        status = f"FPS: {fps:5.1f} | State: {self.robot_state:15s} | Target: {'LOCKED ✓' if self.target_locked else 'SEARCHING'} {faces_indicator}"
        print(f"\r{status}", end='', flush=True)
    
    def run(self):
        """Main robot loop"""
        print()
        
        # Load target
        image_name = input("📂 Target image path (or press Enter to skip): ").strip()
        
        if image_name:
            if not self.load_target(image_name):
                print("❌ Failed to load target. Exiting.")
                return
        else:
            print("⚠️  No target loaded. Robot will not move.")
            print("   (Running in detection-only mode)")
        
        # Initialize camera
        print()
        if not self.camera.initialize():
            print("❌ Camera initialization failed!")
            print("\nTroubleshooting:")
            print("1. Check camera connection")
            print("2. Enable camera: sudo raspi-config -> Interface Options -> Camera")
            print("3. Try: libcamera-hello")
            print("4. Reboot and try again")
            return
        
        print()
        print("🎥 Camera Started!")
        print("🤖 Robot Control Active!")
        print("⌨️  Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                # Read frame
                frame = self.camera.read_frame()
                
                if frame is None:
                    print("\r❌ Failed to read frame!", end='', flush=True)
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Run detection every N frames
                if self.frame_count % DETECTION_INTERVAL == 0:
                    faces = self.detect_faces(frame)
                    
                    # Control robot
                    if self.target_embeddings:
                        self.control_robot(faces)
                    
                    # Save debug frame periodically
                    if SAVE_DEBUG_FRAMES:
                        current_time = time.time()
                        if (current_time - self.last_save_time) > SAVE_INTERVAL:
                            self.save_debug_frame(frame, faces)
                            self.last_save_time = current_time
                
                # Print status
                self.print_status()
                
                # Small delay to prevent CPU overload
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
            self.arduino.close()
            
            avg_fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
            print(f"👋 Stopped. Average FPS: {avg_fps:.1f}")
            print(f"📊 Total frames processed: {self.frame_count}")


if __name__ == "__main__":
    try:
        robot = PrizeRobot()
        robot.run()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
