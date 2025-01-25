import cv2
import numpy as np
from detect import detect
import collections
collections.MutableMapping = collections.abc.MutableMapping
from dronekit import connect, VehicleMode, LocationGlobal
from pymavlink import mavutil
import time
import queue
from utils.general import check_requirements

class DroneObjectTracker:
    def __init__(self, connection_string='/dev/ttyAMA0', baud=921000):
        # Initialize drone connection
        self.vehicle = connect(connection_string, baud=baud, wait_ready=True)
        
        # # Camera setup (adjust parameters for your camera)
        # self.camera = cv2.VideoCapture(0)
        # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # # Initialize object detection model (using YOLOv4-tiny as example)
        # self.net = cv2.dnn.readNet(
        #     "yolov4-tiny.weights",
        #     "yolov4-tiny.cfg"
        # )
        
        # Control parameters
        self.image_center_x = 320  # Half of frame width
        self.image_center_y = 240  # Half of frame height
        self.pid_yaw = PIDController(kp=0.1, ki=0.01, kd=0.05)
        self.pid_pitch = PIDController(kp=0.1, ki=0.01, kd=0.05)
        
        # # Start tracking thread
        # self.tracking_thread = threading.Thread(target=self.track_object)
        # self.tracking_thread.daemon = True
        # self.tracking_thread.start()
        self.track_object()

    def detect_objects(self, frame):
        """
        DEPRECTAED, NOT USED
        Detect objects in frame using YOLO
        """
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        boxes = []
        confidences = []
        class_ids = []

        # Process detections
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], 
                                                   frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def calculate_control_inputs(self, target_box):
        """Calculate required yaw and pitch adjustments"""
        x, y, w, h = target_box
        target_center_x = x + w/2
        target_center_y = y + h/2
        
        # Calculate error from image center
        error_x = self.image_center_x - target_center_x
        error_y = self.image_center_y - target_center_y
        
        # Use PID controllers to get smooth control inputs
        yaw_adjustment = self.pid_yaw.update(error_x)
        pitch_adjustment = self.pid_pitch.update(error_y)
        
        return yaw_adjustment, pitch_adjustment

    def send_control_command(self, yaw_rate, pitch_rate):
        """Send control commands to the drone"""
        # Create the CONDITION_YAW command
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
            0,       # confirmation
            yaw_rate,    # param 1 - yaw rate in deg/s
            25,          # param 2 - yaw speed
            1,           # param 3 - direction: 1: positive, -1: negative
            1,           # param 4 - relative offset 1, absolute angle 0
            0, 0, 0      # param 5-7 not used
        )
        self.vehicle.send_mavlink(msg)
        
        # Send pitch command using ATTITUDE_TARGET
        msg = self.vehicle.message_factory.set_attitude_target_encode(
            0,                                  # time_boot_ms
            1,                                  # target system
            1,                                  # target component
            0b00000000,                        # type mask: enable pitch
            [0, pitch_rate, 0, 0],             # attitude quaternion
            0, 0, 0,                           # roll, pitch, yaw rates
            0.5                                # thrust
        )
        self.vehicle.send_mavlink(msg)

    def track_object(self):
        """Main tracking loop"""
        frameQueue = queue.Queue()
        while True:
            # ret, frame = self.camera.read()
            # if not ret:
            #     continue

            # Detect objects
            # boxes, confidences, class_ids = self.detect_objects(frame)
            check_requirements(exclude=('tensorboard', 'thop'))
            detect(FrameQueue=frameQueue)

            boxes = frameQueue.get()
            
            if boxes:  # If objects detected
                # For simplicity, track the first detected object
                target_box = boxes[0]
                
                # Calculate required adjustments
                yaw_rate, pitch_rate = self.calculate_control_inputs(target_box)
                
                # Send commands to drone
                self.send_control_command(yaw_rate, pitch_rate)
                
                # # Visualize tracking (optional)
                # x, y, w, h = target_box
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.circle(frame, (self.image_center_x, self.image_center_y), 5, (0, 0, 255), -1)
            
            # # Optional: display frame
            # cv2.imshow('Tracking', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    def cleanup(self):
        """Cleanup resources"""
        self.camera.release()
        cv2.destroyAllWindows()
        self.vehicle.close()

class PIDController:
    """Simple PID controller"""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error):
        self.integral += error
        derivative = error - self.previous_error
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        self.previous_error = error
        return output

# Usage example
if __name__ == "__main__":
    # Initialize tracker
    tracker = DroneObjectTracker()
    
    try:
        # Wait for tracking
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        tracker.cleanup()


