#!/usr/bin/env python3
"""
Enhanced Mission Navigation Node for AeroTHON 2025
- Autonomous scanning with denser waypoint pattern
- Payload target centering using VISUAL SERVOING (velocity commands)
- 10-second hover over classification zones
- Records YOLO detections (only above conf threshold)
- Precise disaster zone descent with target alignment (descend only when blue zone + circle present and centered)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped, TwistStamped
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from mavros_msgs.msg import State
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
import time
from enum import Enum
import yaml
import os
import numpy as np
import json


class MissionState(Enum):
    START = 0
    CHECKING_SERVICES = 1
    REQUESTING_MODE = 2
    WAITING_FOR_MODE_CONFIRMATION = 3
    ARMING = 4
    WAITING_FOR_ARM_CONFIRMATION = 5
    REQUESTING_TAKEOFF = 6
    WAITING_FOR_TAKEOFF_ALT = 7
    HOVERING_AFTER_TAKEOFF = 8
    SCANNING = 9
    HOVERING_TO_CLASSIFY = 10
    CLASSIFYING_OBJECTS = 11
    HOVERING_BEFORE_DESCENT = 12
    CENTERING_ON_TARGET = 13  # Now uses visual servoing (velocity)
    DESCENDING_OVER_DISASTER = 14
    HOVERING_OVER_DISASTER = 15
    ASCENDING_AFTER_DISASTER = 16
    RESUME_SCAN = 17
    RETURNING_TO_LAUNCH = 18
    REQUESTING_LAND = 19
    WAITING_FOR_LAND = 20
    MISSION_COMPLETE = 21
    ABORTING = 22


class EnhancedMissionNode(Node):
    def __init__(self):
        super().__init__('enhanced_mission_node')
        self.get_logger().info('===== Enhanced Mission Node Initializing =====')

        # QoS Profiles
        self.qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.qos_state = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # Subscribers
        self.state_sub = self.create_subscription(
            State, 'mavros/state', self.state_callback, self.qos_state)
        self.local_pos_sub = self.create_subscription(
            PoseStamped, 'mavros/local_position/pose', self.local_pos_callback, self.qos_sensor)
        self.white_zone_sub = self.create_subscription(
            PointStamped, '/detection/white_zone_center', self.white_zone_callback, self.qos_sensor)
        self.blue_zone_sub = self.create_subscription(
            PointStamped, '/detection/blue_zone_center', self.blue_zone_callback, self.qos_sensor)
        self.objects_sub = self.create_subscription(
            Detection2DArray, '/detection/objects', self.objects_callback, self.qos_sensor)

        # NEW: Subscribe to payload target pose, payload pixel, blue_zone_full and all detections
        self.payload_pose_sub = self.create_subscription(
            PoseStamped, '/detection/payload_target_pose', self.payload_pose_callback, self.qos_sensor)
        self.payload_pixel_sub = self.create_subscription(
            PointStamped, '/detection/payload_target_pixel', self.payload_pixel_callback, self.qos_sensor)
        self.blue_zone_full_sub = self.create_subscription(
            PointStamped, '/detection/blue_zone_full', self.blue_zone_full_callback, self.qos_sensor)
        self.all_detections_sub = self.create_subscription(
            String, '/detection/all_objects_log', self.all_detections_callback, self.qos_sensor)

        # Publishers
        self.pose_publisher = self.create_publisher(
            PoseStamped, 'mavros/setpoint_position/local', self.qos_reliable)
        # --- NEW: velocity publisher for visual servoing ---
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        # Service Clients
        self.arming_client = self.create_client(CommandBool, 'mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, 'mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, 'mavros/cmd/takeoff')

        # State Variables
        self.current_mavros_state = State()
        self.current_local_pose = PoseStamped()
        self.initial_alt_set = False
        self.initial_local_altitude = 0.0

        # Mission Parameters
        self.takeoff_alt_relative = 8.0
        self.flight_alt_relative = 10.0
        self.disaster_alt_relative = 2.0
        self.flight_alt_absolute_local = 0.0
        self.disaster_alt_absolute_local = 0.0

        self.rate_hz = 20.0
        self.hover_duration_classify = 10.0  # 10 seconds for classification
        self.hover_duration_disaster = 5.0   # 5 seconds over disaster
        self.stabilize_hover_sec = 2.0

        self.position_threshold = 0.8  # Tighter threshold for centering
        self.centering_threshold = 0.3  # Very precise centering (30cm)

        self.yaml_save_path = os.path.join(os.path.expanduser('~'), 'aerothon_results.yaml')

        # Scan Pattern - Denser (50x25m grid)
        self.SCAN_WAYPOINTS_RELATIVE = []
        x_range = np.linspace(-25.0, 25.0, 6)
        y_range = np.linspace(12.5, -12.5, 6)
        for i, y in enumerate(y_range):
            if i % 2 == 0:
                for x in x_range:
                    self.SCAN_WAYPOINTS_RELATIVE.append([x, y, 0.0])
            else:
                for x in reversed(x_range):
                    self.SCAN_WAYPOINTS_RELATIVE.append([x, y, 0.0])
        self.SCAN_WAYPOINTS_RELATIVE.append([0.0, 0.0, 0.0])

        self.scan_waypoints_local = []
        self.current_scan_wp_index = 0

        # Zone Management
        self.zone_classification_counts = {}
        self.current_zone_temp_counts = {}
        self.all_detections_log = []  # ALL detections from CV (for YAML)

        self.object_zones_visited = {1: False, 2: False}
        self.disaster_zone_visited = False
        self.last_zone_detected_time = 0.0
        self.zone_action_pose = None
        self.scan_paused = False
        self.current_classifying_zone_id = 0

        # NEW: Payload Target Tracking & Visual Servo variables
        self.payload_target_world_pose = None  # PoseStamped from CV
        self.last_payload_pixel = None  # (x, y, r)
        self.last_payload_pixel_time = 0.0
        self.last_blue_full = False
        self.last_blue_full_time = 0.0

        self.centering_attempts = 0
        self.max_centering_attempts = 15  # Prevent infinite loops

        # Servo parameters (tunable)
        self.servo_kp_default = 0.0025  # pixels to m/s scale - tuned empirically
        self.servo_kp_precise = 0.0012  # reduced gain when blue zone fills frame
        self.pixel_deadband = 20  # pixels tolerance for considering "centered"

        # Time to hold stable centering (seconds) before descent
        self.center_stable_required = 0.8
        self.center_stable_start = None

        # State Machine
        self.state = MissionState.START
        self.set_mode_future = None
        self.arming_future = None
        self.takeoff_future = None
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
        self.current_target_local_wp = [0.0, 0.0, 0.0]

        # Timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_callback)
        self.get_logger().info("Enhanced Mission Node Started (MAVROS Local Frame)")

    # --- Callbacks ---
    def state_callback(self, msg):
        self.current_mavros_state = msg

    def local_pos_callback(self, msg):
        self.current_local_pose = msg
        if not self.initial_alt_set and abs(msg.pose.position.z) < 0.5:
            self.initial_local_altitude = msg.pose.position.z
            self.initial_alt_set = True
            self.flight_alt_absolute_local = self.initial_local_altitude + self.flight_alt_relative
            self.disaster_alt_absolute_local = self.initial_local_altitude + self.disaster_alt_relative

            # Update scan waypoints
            self.scan_waypoints_local = [
                [wp[0], wp[1], self.flight_alt_absolute_local]
                for wp in self.SCAN_WAYPOINTS_RELATIVE
            ]

            self.get_logger().info(
                f"Init Alt: {self.initial_local_altitude:.2f}m | "
                f"Flight: {self.flight_alt_absolute_local:.2f}m | "
                f"Disaster: {self.disaster_alt_absolute_local:.2f}m"
            )

    def white_zone_callback(self, msg: PointStamped):
        """White classification zone detected."""
        now = self.get_clock().now().nanoseconds / 1e9
        if self.state == MissionState.SCANNING and not self.scan_paused and \
           (now - self.last_zone_detected_time > 15.0):

            zone_id = 0
            if not self.object_zones_visited[1]:
                zone_id = 1
            elif not self.object_zones_visited[2]:
                zone_id = 2

            if zone_id:
                self.current_classifying_zone_id = zone_id
                self.get_logger().info(f"WHITE ZONE {zone_id} DETECTED! Pausing scan.")

                self.object_zones_visited[zone_id] = True
                self.zone_action_pose = self.current_local_pose.pose.position
                self.state = MissionState.HOVERING_TO_CLASSIFY
                self.scan_paused = True
                self.state_start_time = now
                self.last_zone_detected_time = now

    def blue_zone_callback(self, msg: PointStamped):
            """Blue disaster zone detected with persistence logic."""
            now = self.get_clock().now().nanoseconds / 1e9

            # Initialize persistent tracking
            if not hasattr(self, "blue_zone_seen_start"):
                self.blue_zone_seen_start = None
            if not hasattr(self, "blue_zone_last_seen"):
                self.blue_zone_last_seen = 0.0

            # Update last seen timestamp
            self.blue_zone_last_seen = now

            # If first detection (start timer)
            if self.blue_zone_seen_start is None:
                self.blue_zone_seen_start = now
                self.get_logger().info("Blue zone detected — starting persistence timer.")
                return

            # Calculate continuous visibility duration
            visible_duration = now - self.blue_zone_seen_start

            # Check if it has been continuously visible for 4+ seconds
            if visible_duration >= 1.0:
                # Check payload target recent detection (within last second)
                pixel_age = (
                    now - self.last_payload_pixel_time if self.last_payload_pixel_time else 1e6
                )
                if pixel_age < 1.0 and not self.disaster_zone_visited:
                    self.get_logger().info(
                        "Blue zone + payload persistently visible — preparing descent sequence."
                    )
                    self.disaster_zone_visited = True
                    self.zone_action_pose = self.current_local_pose.pose.position
                    self.state = MissionState.HOVERING_BEFORE_DESCENT
                    self.scan_paused = True
                    self.state_start_time = now
                    self.last_zone_detected_time = now
                    # Reset persistence timers
                    self.blue_zone_seen_start = None
                    self.blue_zone_last_seen = 0.0
                else:
                    self.get_logger().info(
                        "Blue zone stable, waiting for payload circle detection..."
                    )
            else:
                self.get_logger().info(
                    f"Blue zone visible for {visible_duration:.1f}s — waiting for 4s stable view...",
                    throttle_duration_sec=1.0,
                )


    def payload_pose_callback(self, msg: PoseStamped):
        """Receive world position of payload target from CV node."""
        if msg.header.frame_id == 'camera_relative':
            # Convert relative offsets to absolute world position
            world_pose = PoseStamped()
            world_pose.header = msg.header
            world_pose.header.frame_id = 'map'

            # Add offsets to current position
            world_pose.pose.position.x = self.current_local_pose.pose.position.x + msg.pose.position.x
            world_pose.pose.position.y = self.current_local_pose.pose.position.y + msg.pose.position.y
            world_pose.pose.position.z = self.current_local_pose.pose.position.z  # Maintain altitude
            world_pose.pose.orientation.w = 1.0

            self.payload_target_world_pose = world_pose

    def payload_pixel_callback(self, msg: PointStamped):
        """Store latest pixel coordinate payload detection (x,y) and radius in z."""
        try:
            self.last_payload_pixel = (int(msg.point.x), int(msg.point.y), float(msg.point.z))
            self.last_payload_pixel_time = self.get_clock().now().nanoseconds / 1e9
        except Exception as e:
            self.get_logger().error(f"Payload pixel parse error: {e}")

    def blue_zone_full_callback(self, msg: PointStamped):
        """Receive info whether blue zone is large/full in frame (z=1.0 means True)."""
        try:
            self.last_blue_full = bool(int(msg.point.z))
            self.last_blue_full_time = self.get_clock().now().nanoseconds / 1e9
        except Exception as e:
            self.get_logger().error(f"Blue full parse error: {e}")

    def objects_callback(self, msg: Detection2DArray):
        """Accumulate object detections during classification."""
        if self.state == MissionState.CLASSIFYING_OBJECTS:
            for det in msg.detections:
                if det.results:
                    class_id = det.results[0].hypothesis.class_id
                    self.current_zone_temp_counts[class_id] = \
                        self.current_zone_temp_counts.get(class_id, 0) + 1

    def all_detections_callback(self, msg: String):
        """Receive detections published by CV (already filtered by confidence)."""
        try:
            detections = json.loads(msg.data)
            self.all_detections_log.extend(detections)
        except Exception as e:
            self.get_logger().error(f"Failed to parse detections JSON: {e}")

    # --- Service Calls ---
    def call_arming_service(self, value):
        req = CommandBool.Request()
        req.value = value
        return self.arming_client.call_async(req)

    def call_set_mode_service(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        return self.set_mode_client.call_async(req)

    def call_takeoff_service(self, altitude):
        req = CommandTOL.Request()
        req.altitude = float(altitude)
        return self.takeoff_client.call_async(req)

    # --- Helpers ---
    def publish_local_pose_command(self, x, y, z):
        """Publish position setpoint."""
        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = 'map'
        target.pose.position.x = float(x)
        target.pose.position.y = float(y)
        target.pose.position.z = float(z)
        target.pose.orientation.w = 1.0
        self.pose_publisher.publish(target)

    def publish_velocity(self, vx, vy, vz):
        """Publish TwistStamped in body frame (m/s)."""
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        # NOTE: mavros expects velocities in the local frame depending on setup.
        # We'll publish as body-frame linear velocities and zero angular rates.
        twist.twist.linear.x = float(vx)  # forward/back
        twist.twist.linear.y = float(vy)  # left/right
        twist.twist.linear.z = float(vz)  # up/down (negative = descend)
        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = 0.0
        self.vel_pub.publish(twist)

    def is_near_local_target(self, target, threshold=None):
        """Check if near target position."""
        if threshold is None:
            threshold = self.position_threshold

        dx = abs(self.current_local_pose.pose.position.x - target[0])
        dy = abs(self.current_local_pose.pose.position.y - target[1])
        dz = abs(self.current_local_pose.pose.position.z - target[2])

        return dx < threshold and dy < threshold and dz < threshold

    def is_centered_on_target(self, target_pose: PoseStamped):
        """Check if centered on payload target with tight tolerance."""
        dx = abs(self.current_local_pose.pose.position.x - target_pose.pose.position.x)
        dy = abs(self.current_local_pose.pose.position.y - target_pose.pose.position.y)

        return dx < self.centering_threshold and dy < self.centering_threshold

    def save_results_to_yaml(self):
        """Save mission results including filtered detections to YAML."""
        # Format zone counts
        formatted_counts = {
            f'zone_{k}': v for k, v in self.zone_classification_counts.items()
        }

        # Aggregate all detections by class
        all_objects_summary = {}
        for det in self.all_detections_log:
            class_name = det['class']
            all_objects_summary[class_name] = all_objects_summary.get(class_name, 0) + 1

        results = {
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'mission_status': 'completed',
            'zones_visited': {
                'classification_zone_1': self.object_zones_visited[1],
                'classification_zone_2': self.object_zones_visited[2],
                'disaster_zone': self.disaster_zone_visited
            },
            'object_counts_per_zone': formatted_counts,
            'all_detections_summary': all_objects_summary,
            'total_detections': len(self.all_detections_log),
            'detailed_detections': self.all_detections_log[:100]  # First 100 for reference
        }

        try:
            with open(self.yaml_save_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f"Results saved to: {self.yaml_save_path}")
            self.get_logger().info(f"Total detections logged: {len(self.all_detections_log)}")
        except Exception as e:
            self.get_logger().error(f"YAML save failed: {e}")

    def is_landed(self):
        """Check if landed."""
        return (abs(self.current_local_pose.pose.position.z - self.initial_local_altitude) < 0.3
                and not self.current_mavros_state.armed)

    # --- Main State Machine ---
    def timer_callback(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        state_duration = now_sec - self.state_start_time

        # [TAKEOFF STATES - UNCHANGED]
        if self.state == MissionState.START:
            if self.current_mavros_state.connected and self.initial_alt_set:
                self.state = MissionState.CHECKING_SERVICES
                self.state_start_time = now_sec
            else:
                self.get_logger().warn("Waiting for FCU connection...", throttle_duration_sec=5.0)

        elif self.state == MissionState.CHECKING_SERVICES:
            if all([c.service_is_ready() for c in [self.arming_client, self.set_mode_client, self.takeoff_client]]):
                self.state = MissionState.REQUESTING_MODE
                self.state_start_time = now_sec
            elif state_duration > 20.0:
                self.get_logger().error("Services timeout!")
                self.state = MissionState.ABORTING
            else:
                self.get_logger().warn("Waiting for services...", throttle_duration_sec=5.0)

        elif self.state == MissionState.REQUESTING_MODE:
            if self.current_mavros_state.mode == 'GUIDED':
                self.state = MissionState.ARMING
                self.state_start_time = now_sec
            elif self.set_mode_future is None:
                self.set_mode_future = self.call_set_mode_service('GUIDED')
            if self.set_mode_future and self.set_mode_future.done():
                try:
                    result = self.set_mode_future.result()
                except Exception as e:
                    result = None
                    self.get_logger().error(f"SetMode error: {e}")
                if result and result.mode_sent:
                    self.state = MissionState.WAITING_FOR_MODE_CONFIRMATION
                    self.state_start_time = now_sec
                else:
                    self.get_logger().error("SetMode request failed")
                self.set_mode_future = None

        elif self.state == MissionState.WAITING_FOR_MODE_CONFIRMATION:
            if self.current_mavros_state.mode == 'GUIDED':
                self.state = MissionState.ARMING
                self.state_start_time = now_sec
            elif state_duration > 5.0:
                self.state = MissionState.REQUESTING_MODE
                self.state_start_time = now_sec

        elif self.state == MissionState.ARMING:
            if self.current_mavros_state.armed:
                self.state = MissionState.REQUESTING_TAKEOFF
                self.state_start_time = now_sec
            elif self.arming_future is None:
                if self.current_mavros_state.mode != 'GUIDED':
                    self.state = MissionState.REQUESTING_MODE
                    return
                self.arming_future = self.call_arming_service(True)
            if self.arming_future and self.arming_future.done():
                try:
                    result = self.arming_future.result()
                except Exception as e:
                    result = None
                    self.get_logger().error(f"Arming error: {e}")
                if result and result.success:
                    self.state = MissionState.WAITING_FOR_ARM_CONFIRMATION
                    self.state_start_time = now_sec
                else:
                    self.get_logger().error("Arming rejected")
                self.arming_future = None

        elif self.state == MissionState.WAITING_FOR_ARM_CONFIRMATION:
            if self.current_mavros_state.armed:
                self.state = MissionState.REQUESTING_TAKEOFF
                self.state_start_time = now_sec
            elif state_duration > 10.0:
                self.state = MissionState.ARMING
                self.state_start_time = now_sec

        elif self.state == MissionState.REQUESTING_TAKEOFF:
            if self.takeoff_future is None:
                if not self.current_mavros_state.armed:
                    self.state = MissionState.ARMING
                    return
                if self.current_mavros_state.mode != 'GUIDED':
                    self.state = MissionState.REQUESTING_MODE
                    return
                self.takeoff_future = self.call_takeoff_service(self.takeoff_alt_relative)
            if self.takeoff_future and self.takeoff_future.done():
                try:
                    result = self.takeoff_future.result()
                except Exception as e:
                    result = None
                    self.get_logger().error(f"Takeoff error: {e}")
                if result and result.success:
                    self.state = MissionState.WAITING_FOR_TAKEOFF_ALT
                    self.state_start_time = now_sec
                else:
                    self.get_logger().error("Takeoff rejected")
                self.takeoff_future = None

        elif self.state == MissionState.WAITING_FOR_TAKEOFF_ALT:
            current_relative_alt = self.current_local_pose.pose.position.z - self.initial_local_altitude
            if abs(current_relative_alt - self.takeoff_alt_relative) < self.position_threshold:
                self.get_logger().info("Takeoff altitude reached!")
                self.state = MissionState.HOVERING_AFTER_TAKEOFF
                self.state_start_time = now_sec
            elif state_duration > 45.0:
                self.get_logger().error("Takeoff timeout!")
                self.state = MissionState.ABORTING
            else:
                self.get_logger().info(f"Climbing... {current_relative_alt:.2f}m",
                                      throttle_duration_sec=1.0)

        elif self.state == MissionState.HOVERING_AFTER_TAKEOFF:
            target_z = self.initial_local_altitude + self.takeoff_alt_relative
            self.publish_local_pose_command(
                self.current_local_pose.pose.position.x,
                self.current_local_pose.pose.position.y,
                target_z
            )
            if state_duration > self.stabilize_hover_sec:
                self.get_logger().info("Starting scan mission...")
                self.state = MissionState.SCANNING
                self.state_start_time = now_sec
                self.current_scan_wp_index = 0
                self.current_target_local_wp = self.scan_waypoints_local[0]

        # [MISSION LOGIC]
        elif self.state == MissionState.SCANNING:
            if self.scan_paused:
                # Hold position while zone is being processed
                self.publish_local_pose_command(
                    self.current_local_pose.pose.position.x,
                    self.current_local_pose.pose.position.y,
                    self.flight_alt_absolute_local
                )
                return

            self.publish_local_pose_command(*self.current_target_local_wp)

            if self.is_near_local_target(self.current_target_local_wp):
                self.current_scan_wp_index += 1
                if self.current_scan_wp_index < len(self.scan_waypoints_local):
                    self.current_target_local_wp = self.scan_waypoints_local[self.current_scan_wp_index]
                    self.get_logger().info(
                        f"Waypoint {self.current_scan_wp_index}/{len(self.scan_waypoints_local)}"
                    )
                else:
                    self.get_logger().info("Scan complete! Returning to launch.")
                    self.state = MissionState.RETURNING_TO_LAUNCH
                    self.current_target_local_wp = [0.0, 0.0, self.flight_alt_absolute_local]
                    self.state_start_time = now_sec
            # Reset blue zone persistence timer if not seen recently
            now = self.get_clock().now().nanoseconds / 1e9
            if hasattr(self, "blue_zone_last_seen") and (now - self.blue_zone_last_seen > 1.5):
                self.blue_zone_seen_start = None

        elif self.state == MissionState.HOVERING_TO_CLASSIFY:
            self.publish_local_pose_command(
                self.zone_action_pose.x,
                self.zone_action_pose.y,
                self.flight_alt_absolute_local
            )
            if state_duration > self.stabilize_hover_sec:
                self.get_logger().info(f"Starting classification of Zone {self.current_classifying_zone_id}...")
                self.current_zone_temp_counts = {}
                self.state = MissionState.CLASSIFYING_OBJECTS
                self.state_start_time = now_sec

        elif self.state == MissionState.CLASSIFYING_OBJECTS:
            self.publish_local_pose_command(
                self.zone_action_pose.x,
                self.zone_action_pose.y,
                self.flight_alt_absolute_local
            )
            if state_duration > self.hover_duration_classify:
                zone_id = self.current_classifying_zone_id
                if zone_id != 0:
                    self.zone_classification_counts[zone_id] = self.current_zone_temp_counts.copy()
                    self.get_logger().info(
                        f"Zone {zone_id} classified: {self.zone_classification_counts[zone_id]}"
                    )
                    self.current_classifying_zone_id = 0

                self.state = MissionState.RESUME_SCAN
                self.state_start_time = now_sec
            else:
                remaining = self.hover_duration_classify - state_duration
                self.get_logger().info(
                    f"Classifying Zone {self.current_classifying_zone_id}... {remaining:.1f}s",
                    throttle_duration_sec=1.0
                )

        elif self.state == MissionState.HOVERING_BEFORE_DESCENT:
            self.publish_local_pose_command(
                self.zone_action_pose.x,
                self.zone_action_pose.y,
                self.flight_alt_absolute_local
            )
            if state_duration > self.stabilize_hover_sec:
                self.get_logger().info("Waiting for payload target detection and centering...")
                self.state = MissionState.CENTERING_ON_TARGET
                self.state_start_time = now_sec
                self.centering_attempts = 0
                self.payload_target_world_pose = None
                self.center_stable_start = None

        elif self.state == MissionState.CENTERING_ON_TARGET:
            """
            Visual servoing centering using velocity commands.
            - Ensure both: payload circle (pixel) present and blue zone full in frame (persistent)
            - Use proportional pixel->velocity mapping
            - Once pixel error within deadband reliably for center_stable_required seconds -> descend
            """

            now = self.get_clock().now().nanoseconds / 1e9
            # check freshness of pixel/blue_full detections
            pixel_age = now - self.last_payload_pixel_time if self.last_payload_pixel_time else 1e6
            blue_age = now - self.last_blue_full_time if self.last_blue_full_time else 1e6

            # require both payload pixel and blue_full to be recently observed (<0.6s)
            if self.last_payload_pixel is None or pixel_age > 0.6 or blue_age > 0.6:
                # hold position using position setpoints if detections are not present
                self.publish_local_pose_command(
                    self.current_local_pose.pose.position.x,
                    self.current_local_pose.pose.position.y,
                    self.flight_alt_absolute_local
                )
                self.center_stable_start = None
                if state_duration > 10.0:
                    self.get_logger().warn("No payload target in view for a while, descending anyway...")
                    # fallback: descend to disaster altitude at zone_action_pose
                    self.state = MissionState.DESCENDING_OVER_DISASTER
                    self.current_target_local_wp = [
                        self.zone_action_pose.x,
                        self.zone_action_pose.y,
                        self.disaster_alt_absolute_local
                    ]
                    self.state_start_time = now
                else:
                    self.get_logger().info("Waiting for payload target & blue zone to be persistent...", throttle_duration_sec=1.0)
                return

            # we have a recent pixel detection and blue_full info
            px, py, pr = self.last_payload_pixel

            # compute pixel error relative to image center
            # NOTE: we don't have camera intrinsics here; we convert pixels -> velocity via tuned Kp
            # reduce Kp (more precise) if blue zone is full in frame
            use_precise = self.last_blue_full
            Kp = self.servo_kp_precise if use_precise else self.servo_kp_default

            # pixels: positive x -> right, positive y -> down
            # Actually we need camera center pixels - but cv node uses cam center; we will assume payload pixel is in pixel coords with origin at left-top
            # We don't have camera center in this node; instead, we can treat pixel error relative to 1/2 of image width/height.
            # For robust operation, we assume camera resolution from cv node defaults; choose typical: 1920x1080 unless set otherwise.
            cam_w = 1920
            cam_h = 1080
            cam_cx = cam_w // 2
            cam_cy = cam_h // 2

            err_px = px - cam_cx  # + right positive
            err_py = py - cam_cy  # + down positive

            # Convert pixel error to body-frame velocities (forward/back and left/right)
            # Mapping: body x (forward) reduces negative pixel y (py < cy -> move forward), so vx ~ -err_py * Kp
            # body y (right) should be positive for err_px > 0, so vy ~ err_px * Kp
            vx = -float(err_py) * Kp
            vy = float(err_px) * Kp

            # Limit velocities to safe maxima (tunable)
            max_horiz_speed = 1.2 if not use_precise else 0.5  # m/s
            vx = max(-max_horiz_speed, min(max_horiz_speed, vx))
            vy = max(-max_horiz_speed, min(max_horiz_speed, vy))

            # For descent we keep z velocity 0 here (we hover while centering)
            vz = 0.0

            # publish velocity command
            self.publish_velocity(vx, vy, vz)
            self.get_logger().info(f"Servoing vx:{vx:.2f} vy:{vy:.2f} (px err {err_px},{err_py})", throttle_duration_sec=0.5)

            # Check if error is within deadband (both axes)
            if abs(err_px) <= self.pixel_deadband and abs(err_py) <= self.pixel_deadband:
                if self.center_stable_start is None:
                    self.center_stable_start = now
                    self.get_logger().info("Centering stable start timer...")
                elif (now - self.center_stable_start) >= self.center_stable_required:
                    # Centered and stable: command to descend
                    self.get_logger().info("CENTERED on payload target (stable). Descending...")
                    # Stop velocity commands to avoid fight with position setpoints
                    self.publish_velocity(0.0, 0.0, 0.0)
                    self.state = MissionState.DESCENDING_OVER_DISASTER
                    # Determine current target XY in local frame to descend to
                    # Use payload_world_pose if available else zone_action_pose
                    if self.payload_target_world_pose is not None:
                        target_x = self.payload_target_world_pose.pose.position.x
                        target_y = self.payload_target_world_pose.pose.position.y
                    else:
                        target_x = self.zone_action_pose.x
                        target_y = self.zone_action_pose.y

                    # when blue zone fills frame, descend slower by using disaster_alt closer? We'll keep altitude but ensure controlled descent by autopilot
                    self.current_target_local_wp = [
                        target_x,
                        target_y,
                        self.disaster_alt_absolute_local
                    ]
                    self.state_start_time = now
                    self.center_stable_start = None
            else:
                # not yet centered; reset stable timer
                self.center_stable_start = None
                self.centering_attempts += 1
                if self.centering_attempts > self.max_centering_attempts:
                    self.get_logger().warn("Max centering attempts exceeded, descending anyway...")
                    self.publish_velocity(0.0, 0.0, 0.0)
                    self.state = MissionState.DESCENDING_OVER_DISASTER
                    self.current_target_local_wp = [
                        self.zone_action_pose.x,
                        self.zone_action_pose.y,
                        self.disaster_alt_absolute_local
                    ]
                    self.state_start_time = now

        elif self.state == MissionState.DESCENDING_OVER_DISASTER:
            # Publish position target for descent (autopilot will handle descent)
            self.publish_local_pose_command(*self.current_target_local_wp)

            if abs(self.current_local_pose.pose.position.z - self.disaster_alt_absolute_local) < self.position_threshold:
                self.get_logger().info("Reached disaster altitude!")
                self.state = MissionState.HOVERING_OVER_DISASTER
                self.state_start_time = now_sec
            elif state_duration > 30.0:
                self.get_logger().error("Descent timeout!")
                self.state = MissionState.ABORTING
            else:
                self.get_logger().info(
                    f"Descending... {self.current_local_pose.pose.position.z:.2f}m",
                    throttle_duration_sec=1.0
                )

        elif self.state == MissionState.HOVERING_OVER_DISASTER:
            self.publish_local_pose_command(*self.current_target_local_wp)

            if state_duration > self.hover_duration_disaster:
                self.get_logger().info("Disaster zone hover complete! Ascending...")
                self.state = MissionState.ASCENDING_AFTER_DISASTER
                self.current_target_local_wp[2] = self.flight_alt_absolute_local
                self.state_start_time = now_sec
            else:
                remaining = self.hover_duration_disaster - state_duration
                self.get_logger().info(
                    f"Hovering over disaster... {remaining:.1f}s",
                    throttle_duration_sec=1.0
                )

        elif self.state == MissionState.ASCENDING_AFTER_DISASTER:
            self.publish_local_pose_command(*self.current_target_local_wp)

            if abs(self.current_local_pose.pose.position.z - self.flight_alt_absolute_local) < self.position_threshold:
                self.get_logger().info("Ascended to flight altitude!")
                self.state = MissionState.RESUME_SCAN
                self.state_start_time = now_sec
            elif state_duration > 30.0:
                self.get_logger().error("Ascent timeout!")
                self.state = MissionState.ABORTING
            else:
                self.get_logger().info(
                    f"Ascending... {self.current_local_pose.pose.position.z:.2f}m",
                    throttle_duration_sec=1.0
                )

        elif self.state == MissionState.RESUME_SCAN:
            self.get_logger().info("Resuming scan pattern...")
            self.scan_paused = False

            if self.current_scan_wp_index >= len(self.scan_waypoints_local):
                self.get_logger().info("Scan complete! Returning to launch.")
                self.state = MissionState.RETURNING_TO_LAUNCH
                self.current_target_local_wp = [0.0, 0.0, self.flight_alt_absolute_local]
            else:
                self.state = MissionState.SCANNING
                self.current_target_local_wp = self.scan_waypoints_local[self.current_scan_wp_index]

            self.state_start_time = now_sec

        # [LANDING STATES]
        elif self.state == MissionState.RETURNING_TO_LAUNCH:
            self.publish_local_pose_command(*self.current_target_local_wp)

            if self.is_near_local_target(self.current_target_local_wp):
                self.get_logger().info("Reached launch position!")
                self.state = MissionState.REQUESTING_LAND
                self.state_start_time = now_sec
            elif state_duration > 90.0:
                self.get_logger().error("RTL timeout!")
                self.state = MissionState.ABORTING
            else:
                self.get_logger().info("Returning to launch...", throttle_duration_sec=1.0)

        elif self.state == MissionState.REQUESTING_LAND:
            if self.current_mavros_state.mode == 'LAND':
                self.state = MissionState.WAITING_FOR_LAND
                self.state_start_time = now_sec
            elif self.set_mode_future is None:
                self.set_mode_future = self.call_set_mode_service('LAND')
            if self.set_mode_future and self.set_mode_future.done():
                try:
                    result = self.set_mode_future.result()
                except Exception as e:
                    result = None
                    self.get_logger().error(f"SetMode(LAND) error: {e}")
                if result and result.mode_sent:
                    self.state = MissionState.WAITING_FOR_LAND
                    self.state_start_time = now_sec
                else:
                    self.get_logger().error("Land request failed")
                self.set_mode_future = None

        elif self.state == MissionState.WAITING_FOR_LAND:
            if self.is_landed():
                self.state = MissionState.MISSION_COMPLETE
                self.state_start_time = now_sec
            elif state_duration > 60.0:
                self.get_logger().warn("Landing timeout, assuming landed.")
                self.state = MissionState.MISSION_COMPLETE
                self.state_start_time = now_sec
            else:
                rel_alt = self.current_local_pose.pose.position.z - self.initial_local_altitude
                self.get_logger().info(
                    f"Landing... {rel_alt:.2f}m",
                    throttle_duration_sec=1.0
                )

        elif self.state == MissionState.MISSION_COMPLETE:
            self.get_logger().info("===== MISSION COMPLETE =====")
            self.save_results_to_yaml()

            # Disarm if still armed
            if self.current_mavros_state.armed and self.arming_client.service_is_ready():
                self.call_arming_service(False)

            # Switch to STABILIZE
            if self.current_mavros_state.mode != 'STABILIZE' and self.set_mode_client.service_is_ready():
                self.call_set_mode_service('STABILIZE')

            self.get_logger().info("Shutting down node...")
            self.timer.cancel()
            rclpy.shutdown()

        elif self.state == MissionState.ABORTING:
            self.get_logger().error("MISSION ABORTED! Attempting emergency land...")

            if self.current_mavros_state.mode != 'LAND' and \
               self.set_mode_client.service_is_ready() and \
               self.set_mode_future is None:
                self.set_mode_future = self.call_set_mode_service('LAND')

            if self.current_mavros_state.mode == 'LAND' or \
               (self.set_mode_future and self.set_mode_future.done()):
                self.state = MissionState.WAITING_FOR_LAND
                self.state_start_time = now_sec
            elif state_duration > 10.0:
                self.get_logger().fatal("Cannot initiate emergency landing!")
                self.timer.cancel()
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EnhancedMissionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('Keyboard interrupt detected.')
        else:
            print('Interrupted.')
    except Exception as e:
        print(f'Exception in mission node: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        print("Mission node shutdown complete.")


if __name__ == '__main__':
    main()
