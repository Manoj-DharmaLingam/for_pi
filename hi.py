import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from collections import deque
import time
import serial

# Try to import Picamera2
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# ================= CONFIGURATION =================
# Camera Settings - OPTIMIZED FOR SPEED
USE_PICAMERA = PICAMERA_AVAILABLE
FRAME_WIDTH = 480  # Reduced from 640 for speed
FRAME_HEIGHT = 360  # Reduced from 480 for speed
CAMERA_FPS = 30
USB_CAMERA_ID = 0

# Display Settings
ENABLE_DISPLAY = True
DISPLAY_WINDOW_NAME = "Prize Robot"

# Serial Communication
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
ENABLE_SERIAL = True

# Model Settings - BALANCED (not too heavy, not too light)
MODEL_NAME = 'buffalo_s'  # Lighter model for speed
CTX_ID = -1

# Detection Settings - SPEED OPTIMIZED
DET_SIZE = (320, 320)  # Good balance
DET_THRESH = 0.45  # Balanced threshold

# Recognition Settings - OPTIMIZED
RECOGNITION_THRESHOLD = 0.35  # Balanced
MIN_FACE_SIZE = 30
SMOOTH_FRAMES = 3  # Reduced from 5

# Performance Settings - MAXIMUM SPEED
MAX_FACES_TO_PROCESS = 3  # Reduced from 10
DETECTION_INTERVAL = 5  # Increased from 2 (detect less often)
SKIP_FRAME_PROCESSING = True  # Skip every other frame

# Robot Control
TARGET_CENTER_TOLERANCE = 70
MOVEMENT_UPDATE_INTERVAL = 0.4
TARGET_SIZE_THRESHOLD = 200

# Debug Settings
SAVE_DEBUG_FRAMES = False  # Disabled for speed
SAVE_INTERVAL = 60
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
            except:
                return False
        else:
            cmd_name = {
                CMD_FORWARD: "FWD", CMD_BACKWARD: "BCK",
                CMD_LEFT: "LFT", CMD_RIGHT: "RGT",
                CMD_STOP: "STP", CMD_FORWARD_SLOW: "FSL"
            }.get(command, command.strip())
            print(f"🤖 {cmd_name}")
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


class CameraHandler:
    def __init__(self):
        self.camera = None
        self.camera_type = None
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        
    def initialize(self):
        # Try PiCamera
        if USE_PICAMERA and PICAMERA_AVAILABLE:
            try:
                print("📹 Initializing Pi Camera...")
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
                    print("✅ Pi Camera ready!")
                    return True
            except Exception as e:
                print(f"❌ Pi Camera failed: {e}")
                if self.camera:
                    try:
                        self.camera.stop()
                    except:
                        pass
                self.camera = None
        
        # Try USB Camera
        print("📹 Initializing USB Camera...")
        try:
            self.camera = cv2.VideoCapture(USB_CAMERA_ID)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize lag
            
            ret, test_frame = self.camera.read()
            if ret and test_frame is not None:
                self.camera_type = "usb"
                print("✅ USB Camera ready!")
                return True
        except Exception as e:
            print(f"❌ USB Camera failed: {e}")
        
        return False
    
    def read_frame(self):
        if self.camera is None:
            return None
        
        try:
            if self.camera_type == "picamera":
                frame = self.camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            elif self.camera_type == "usb":
                ret, frame = self.camera.read()
                return frame if ret else None
        except:
            return None
        
        return None
    
    def stop(self):
        if self.camera:
            try:
                if self.camera_type == "picamera":
                    self.camera.stop()
                elif self.camera_type == "usb":
                    self.camera.release()
            except:
                pass


class FastPrizeRobot:
    """Optimized Prize Robot - High Speed Version"""
    
    def __init__(self):
        print("="*60)
        print("  🚀 Prize Robot - SPEED OPTIMIZED")
        print("="*60)
        
        if SAVE_DEBUG_FRAMES:
            os.makedirs(DEBUG_FOLDER, exist_ok=True)
        
        self.arduino = ArduinoController()
        self.camera = CameraHandler()
        
        # Fast model loading
        print("🧠 Loading AI model (30-45 seconds)...")
        try:
            self.app = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE, det_thresh=DET_THRESH)
            print("✅ AI model loaded!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        # State
        self.target_embeddings = []
        self.frame_count = 0
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
        self.last_detection_time = 0
        
        self.target_locked = False
        self.target_bbox = None
        self.last_faces = []  # Cache last detection
        self.frame_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
        self.robot_state = "IDLE"
        
        # Quick recognition
        self.recognition_buffer = deque(maxlen=SMOOTH_FRAMES)
        self.last_similarity = 0.0
        
        print("✅ Robot ready!")
        print("="*60)
    
    def load_target(self, image_path):
        """Fast target loading - fewer embeddings for speed"""
        if not os.path.exists(image_path):
            print(f"⚠️  File not found: {image_path}")
            return False
        
        print(f"🎯 Loading target...")
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Could not read image")
            return False
        
        embeddings = []
        
        # Original
        faces = self.app.get(img)
        if len(faces) > 0:
            embeddings.append(faces[0].embedding)
            print("   ✓ Original")
        
        # Flipped
        flipped = cv2.flip(img, 1)
        faces = self.app.get(flipped)
        if len(faces) > 0:
            embeddings.append(faces[0].embedding)
            print("   ✓ Flipped")
        
        # Quick brightness variations (only 4)
        for gamma in [0.7, 0.9, 1.1, 1.3]:
            adjusted = self.adjust_gamma(img, gamma)
            faces = self.app.get(adjusted)
            if len(faces) > 0:
                embeddings.append(faces[0].embedding)
        print(f"   ✓ Brightness x4")
        
        if len(embeddings) == 0:
            print("❌ No face found")
            return False
        
        self.target_embeddings = embeddings
        print(f"✅ Loaded! {len(embeddings)} embeddings")
        return True
    
    @staticmethod
    def adjust_gamma(image, gamma):
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
        
        # Fast similarity check (just max)
        similarities = [self.compute_similarity(ref, embedding) 
                       for ref in self.target_embeddings]
        max_sim = max(similarities)
        
        # Simple smoothing
        self.recognition_buffer.append(max_sim > RECOGNITION_THRESHOLD)
        
        if len(self.recognition_buffer) >= SMOOTH_FRAMES:
            matches = sum(list(self.recognition_buffer)[-SMOOTH_FRAMES:])
            if matches >= 2:  # 2 out of 3
                return True, max_sim
        
        return max_sim > RECOGNITION_THRESHOLD, max_sim
    
    def detect_faces(self, frame):
        """Fast detection"""
        try:
            faces = self.app.get(frame)
            
            # Quick filter
            valid = [f for f in faces if (f.bbox[2] - f.bbox[0]) >= MIN_FACE_SIZE]
            
            if len(valid) > MAX_FACES_TO_PROCESS:
                valid.sort(key=lambda x: x.det_score, reverse=True)
                valid = valid[:MAX_FACES_TO_PROCESS]
            
            return valid
        except:
            return []
    
    def calculate_movement_command(self, target_bbox):
        if target_bbox is None:
            return CMD_STOP, "NO_TARGET"
        
        x1, y1, x2, y2 = target_bbox.astype(int)
        center_x = (x1 + x2) // 2
        offset_x = center_x - self.frame_center[0]
        
        size = ((x2 - x1) + (y2 - y1)) / 2
        
        if abs(offset_x) > TARGET_CENTER_TOLERANCE:
            return (CMD_RIGHT, "ALIGN_R") if offset_x > 0 else (CMD_LEFT, "ALIGN_L")
        
        if size > TARGET_SIZE_THRESHOLD:
            return CMD_STOP, "REACHED"
        
        return CMD_FORWARD_SLOW, "APPROACH"
    
    def control_robot(self, faces):
        target_found = False
        best_bbox = None
        best_sim = 0.0
        
        for face in faces:
            if self.target_embeddings:
                is_match, sim = self.is_target_match(face.embedding)
                if is_match and sim > best_sim:
                    target_found = True
                    best_bbox = face.bbox
                    best_sim = sim
        
        self.last_similarity = best_sim
        
        if target_found:
            self.target_locked = True
            self.target_bbox = best_bbox
            cmd, state = self.calculate_movement_command(best_bbox)
            self.robot_state = state
            self.arduino.send_command(cmd)
        else:
            if self.target_locked:
                self.arduino.send_command(CMD_STOP)
                self.robot_state = "LOST"
            else:
                self.robot_state = "SEARCH"
            self.target_locked = False
            self.target_bbox = None
    
    def draw_fast_display(self, frame):
        """Minimal drawing for speed"""
        display = frame.copy()
        
        # Draw only target face (skip others for speed)
        if self.target_bbox is not None:
            box = self.target_bbox.astype(int)
            
            # Green box for target
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Simple crosshair
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            cv2.line(display, (cx-15, cy), (cx+15, cy), (0, 255, 255), 2)
            cv2.line(display, (cx, cy-15), (cx, cy+15), (0, 255, 255), 2)
            
            # Label
            label = f"{self.last_similarity:.2f}"
            cv2.putText(display, label, (box[0], box[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Minimal status (top-left)
        fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        status_color = (0, 255, 0) if self.target_locked else (0, 0, 255)
        
        cv2.putText(display, f"FPS:{fps:.0f}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, self.robot_state, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        return display
    
    def update_fps(self):
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)
    
    def run(self):
        """Fast main loop"""
        print()
        image = input("📂 Target image (or Enter to skip): ").strip()
        
        if image:
            if not self.load_target(image):
                return
        else:
            print("⚠️  No target - detection only")
        
        print()
        if not self.camera.initialize():
            print("❌ Camera failed!")
            return
        
        if ENABLE_DISPLAY:
            cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(DISPLAY_WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
        
        print()
        print("🎥 Running! (Q=quit, SPACE=stop)")
        print()
        
        try:
            while True:
                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                
                # Skip frames for speed
                if SKIP_FRAME_PROCESSING and self.frame_count % 2 == 0:
                    if ENABLE_DISPLAY:
                        cv2.imshow(DISPLAY_WINDOW_NAME, frame)
                    self.update_fps()
                    continue
                
                # Detect periodically
                current_time = time.time()
                if (current_time - self.last_detection_time) >= (DETECTION_INTERVAL / CAMERA_FPS):
                    faces = self.detect_faces(frame)
                    self.last_faces = faces
                    self.last_detection_time = current_time
                    
                    if self.target_embeddings:
                        self.control_robot(faces)
                else:
                    faces = self.last_faces  # Reuse previous detection
                
                # Display
                if ENABLE_DISPLAY:
                    display = self.draw_fast_display(frame)
                    cv2.imshow(DISPLAY_WINDOW_NAME, display)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        self.arduino.stop()
                
                self.update_fps()
        
        except KeyboardInterrupt:
            print("\n⚠️  Stopped by user")
        
        finally:
            print("\n🧹 Cleanup...")
            self.arduino.stop()
            time.sleep(0.3)
            self.camera.stop()
            if ENABLE_DISPLAY:
                cv2.destroyAllWindows()
            self.arduino.close()
            
            fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
            print(f"👋 Done! Avg FPS: {fps:.1f}")


if __name__ == "__main__":
    try:
        robot = FastPrizeRobot()
        robot.run()
    except KeyboardInterrupt:
        print("\n👋 Bye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
