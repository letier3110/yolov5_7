import time
from dronekit import connect, VehicleMode, LocationGlobal
from pymavlink import mavutil
import math

class DroneHeadingController:
    def __init__(self, connection_string='/dev/ttyAMA0', baud=57600):
        # Connect to the Vehicle
        print('Connecting to vehicle on: %s' % connection_string)
        self.vehicle = connect(connection_string, baud=baud, wait_ready=True)
        
        # Control parameters
        self.max_yaw_rate = 30  # degrees per second
        self.min_yaw_rate = 5   # minimum rate to overcome inertia
        self.heading_tolerance = 5  # degrees
        
        # Initialize previous values for smooth control
        self.last_heading_error = 0
        self.last_command_time = time.time()

    def get_heading_error(self, target_heading):
        """Calculate the shortest angular distance between current and target heading"""
        current_heading = self.vehicle.heading
        error = target_heading - current_heading
        
        # Normalize to [-180, 180]
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
            
        return error

    def send_yaw_command(self, heading_deg, relative=False):
        """
        Send MAVLink command to control vehicle yaw.
        
        heading_deg: Target heading in degrees (0-360)
        relative: False for absolute heading, True for relative to current heading
        """
        if relative:
            is_relative = 1  # yaw relative to direction of travel
        else:
            is_relative = 0  # yaw is an absolute angle

        # Create the CONDITION_YAW command using command_long_encode()
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
            0,       # confirmation
            heading_deg,    # param 1 - target angle
            self.max_yaw_rate,          # param 2 - angular speed
            1,           # param 3 - direction: 1: positive, -1: negative
            is_relative, # param 4 - relative offset 1, absolute angle 0
            0, 0, 0)    # param 5-7 not used
        
        # Send command to vehicle
        self.vehicle.send_mavlink(msg)

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
        
        # Send command to vehicle
        self.vehicle.send_mavlink(msg)

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
            # Get current position
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

    def cleanup(self):
        """Close vehicle connection"""
        self.vehicle.close()

# Usage example
if __name__ == "__main__":
    # Initialize controller
    controller = DroneHeadingController()
    
    try:
        # Example: Point to absolute heading
        controller.control_heading_continuous(90)  # Point to East
        
        # Example: Point at GPS location
        # controller.point_at_target(37.7749, -122.4194)  # Example coordinates
        
    except KeyboardInterrupt:
        print("Control interrupted")
    finally:
        controller.cleanup()
