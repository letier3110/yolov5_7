import cv2
import numpy as np
from detect import detect
import collections
collections.MutableMapping = collections.abc.MutableMapping
from dronekit import connect, VehicleMode, LocationGlobal
from pymavlink import mavutil
import time
import queue
# from utils.general import check_requirements

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

class DroneObjectTracker:
    def __init__(self, connection_string='/dev/ttyAMA0', baud=921000):
        # Initialize drone connection
        self.vehicle = connect(connection_string, baud=baud, wait_ready=True)

        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True

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


    def send_yaw_rate_command(self, yaw_rate_deg_sec):
        """
        Send MAVLink command to control vehicle yaw rate.
        yaw_rate_deg_sec: Target yaw rate in degrees per second
        """

        # Ensure yaw rate is within limits
        yaw_rate = max(min(yaw_rate_deg_sec, self.max_yaw_rate), -self.max_yaw_rate)

        # If yaw rate is very small, increase to minimum or zero
        if 0 < abs(yaw_rate) < self.min_yaw_rate:
            yaw_rate = math.copysign(self.min_yaw_rate, yaw_rate)
        elif abs(yaw_rate) < 0.1:  # Dead band to prevent oscillation
            yaw_rate = 0

        # Create the ATTITUDE_TARGET command
        msg = self.vehicle.message_factory.set_attitude_target_encode(
            0,                                  # time_boot_ms
            1,                                  # target system
            1,                                  # target component
            0b100111,                          # type mask: only yaw rate
            [0, 0, 0, 0],                      # attitude quaternion (not used)
            0, 0, math.radians(yaw_rate),      # roll, pitch, yaw rates
            0.5                                # thrust
        )

        self.vehicle.send_mavlink(msg)


    def get_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate bearing between two GPS coordinates
        Returns bearing in degrees (-180 to +180)
        """

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        d_lon = lon2 - lon1

        y = math.sin(d_lon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - \
            math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        bearing = math.atan2(y, x)

        return math.degrees(bearing)


    def control_heading_continuous(self, target_heading, max_duration=30):
        """
        Continuously control heading using yaw rate commands.
        Implements a basic proportional controller with safety timeout.
        """

        start_time = time.time()

        while True:

            # Safety timeout
            if time.time() - start_time > max_duration:
                print("Maximum control duration exceeded")
                self.send_yaw_rate_command(0)  # Stop rotation
                break

            # Calculate heading error
            error = self.get_heading_error(target_heading)

            # Check if we're within tolerance

            if abs(error) < self.heading_tolerance:
                print("Target heading achieved")
                self.send_yaw_rate_command(0)  # Stop rotation
                break

            # Simple proportional control for yaw rate
            # You might want to implement PID control for better performance
            kp = 0.5  # Proportional gain
            yaw_rate = kp * error

            # Send the command
            self.send_yaw_rate_command(yaw_rate)

            # Small delay to prevent flooding the autopilot
            time.sleep(0.1)


    def point_at_target(self, target_lat, target_lon):
        """
        Calculate and maintain heading to point at a GPS target
        """

        while True:
            current_lat = self.vehicle.location.global_relative_frame.lat
            current_lon = self.vehicle.location.global_relative_frame.lon

            # Calculate bearing to target
            target_bearing = self.get_bearing(current_lat, current_lon,
                                            target_lat, target_lon)

            # Convert bearing to heading (0-360)
            target_heading = target_bearing if target_bearing >= 0 else target_bearing + 360

            # Control heading to target
            self.control_heading_continuous(target_heading, max_duration=5)

            # Small delay before next update
            time.sleep(0.5)


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
        """
        Calculate required yaw and pitch adjustments
        """

        x, y, w, h = target_box.tolist()[0]

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


    def contol_function(self, target_box):

        if target_box == None:
            return False

        # Calculate required adjustments
        yaw_rate, pitch_rate = self.calculate_control_inputs(target_box)

        # Send commands to drone
        self.send_control_command(yaw_rate, pitch_rate)


    def track_object(self):

        # Detect objects
        # boxes, confidences, class_ids = self.detect_objects(frame)
        check_requirements(exclude=('tensorboard', 'thop'))

        detect(self.contol_function)


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
    tracker.point_at_target(37.7749, -122.4194)

    try:
        # Wait for tracking
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # cap.release()
        # cv.destroyAllWindows()
        tracker.cleanup()




