#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Enhanced Perception Node for AeroTHON 2025
- Detects white zones, blue disaster zones, red payload targets
- Records ALL YOLO detections (regardless of confidence) for YAML export
- Publishes payload target position for precise centering
- Optimized for IMX219 8MP camera (1920x1080)
"""
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO
import os

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String
import json


class EnhancedPerceptionNode(Node):
    def __init__(self):
        super().__init__('enhanced_perception_node')
        self.get_logger().info('===== Enhanced Perception Node Initializing =====')

        # --- Parameters ---
        self.declare_parameter('camera_topic', '/camera/camera/image_raw')
        self.declare_parameter('yolo_model_path', 'final_model_best.pt')
        self.declare_parameter('yolo_conf_threshold', 0.1)  # 10% confidence
        #self.declare_parameter('yolo_conf_log_all', 0.0)  # Log everything for YAML
        self.declare_parameter('min_zone_area', 2000.0)
        
        # Camera geometry - IMX219 default 1920x1080
        self.declare_parameter('camera_width', 1920)
        self.declare_parameter('camera_height', 1080)
        self.declare_parameter('camera_fov_horizontal_deg', 62.2)  # IMX219 typical
        
        # HSV/BGR Thresholds
        self.declare_parameter('white_lower_bgr', [190, 190, 190])
        self.declare_parameter('white_upper_bgr', [255, 255, 255])
        self.declare_parameter('blue_lower_hsv', [90, 80, 80])
        self.declare_parameter('blue_upper_hsv', [135, 255, 255])
        self.declare_parameter('red_lower1_hsv', [0, 100, 80])
        self.declare_parameter('red_upper1_hsv', [10, 255, 255])
        self.declare_parameter('red_lower2_hsv', [165, 100, 80])
        self.declare_parameter('red_upper2_hsv', [180, 255, 255])

        # Get parameters
        camera_topic = self.get_parameter('camera_topic').value
        model_path = self.get_parameter('yolo_model_path').value
        self.yolo_conf = self.get_parameter('yolo_conf_threshold').value
        #self.yolo_conf_log = self.get_parameter('yolo_conf_log_all').value
        self.min_zone_area = self.get_parameter('min_zone_area').value
        
        self.cam_width = self.get_parameter('camera_width').value
        self.cam_height = self.get_parameter('camera_height').value
        self.cam_center_x = self.cam_width // 2
        self.cam_center_y = self.cam_height // 2
        self.cam_fov_h = self.get_parameter('camera_fov_horizontal_deg').value
        
        # Load thresholds
        self.white_lower_bgr = np.array(self.get_parameter('white_lower_bgr').value)
        self.white_upper_bgr = np.array(self.get_parameter('white_upper_bgr').value)
        self.blue_lower_hsv = np.array(self.get_parameter('blue_lower_hsv').value)
        self.blue_upper_hsv = np.array(self.get_parameter('blue_upper_hsv').value)
        self.red_lower1 = np.array(self.get_parameter('red_lower1_hsv').value)
        self.red_upper1 = np.array(self.get_parameter('red_upper1_hsv').value)
        self.red_lower2 = np.array(self.get_parameter('red_lower2_hsv').value)
        self.red_upper2 = np.array(self.get_parameter('red_upper2_hsv').value)

        self.morph_kernel = np.ones((5, 5), np.uint8)
        self.bridge = CvBridge()

        # YOLO Model
        self.model = None
        if not os.path.exists(model_path):
            self.get_logger().error(f"YOLO model NOT FOUND: {os.path.abspath(model_path)}")
        else:
            try:
                self.model = YOLO(model_path)
                self.get_logger().info(f'YOLO loaded: {model_path}')
            except Exception as e:
                self.get_logger().error(f'YOLO loading FAILED: {e}')

        # Subscribers
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.local_pos_sub = self.create_subscription(
            PoseStamped, 'mavros/local_position/pose', self.local_pos_callback, 10)
        
        self.get_logger().info(f'Subscribed to: {camera_topic}')

        # Publishers
        self.white_zone_pub = self.create_publisher(PointStamped, '/detection/white_zone_center', 10)
        self.blue_zone_pub = self.create_publisher(PointStamped, '/detection/blue_zone_center', 10)
        self.objects_pub = self.create_publisher(Detection2DArray, '/detection/objects', 10)
        
        # NEW: Payload target as PoseStamped (world position for navigation)
        self.payload_target_pose_pub = self.create_publisher(
            PoseStamped, '/detection/payload_target_pose', 10)
        
        # NEW: Payload target pixel coordinates (for velocity control backup)
        self.payload_target_pixel_pub = self.create_publisher(
            PointStamped, '/detection/payload_target_pixel', 10)
        
        # NEW: All detections for YAML logging (JSON string)
        self.all_detections_pub = self.create_publisher(
            String, '/detection/all_objects_log', 10)
        
        self.debug_image_pub = self.create_publisher(Image, '/detection/debug_image', 10)

        # State tracking
        self.current_altitude = 0.0  # For pixel-to-world conversion
        self.last_payload_target_pixel = None  # Store for navigation
        
        # Detection logging
        self.all_detections_buffer = []  # Accumulate ALL detections

        # Debug window
        self.debug_window_name = "Enhanced Perception Debug"
        cv2.namedWindow(self.debug_window_name, cv2.WINDOW_AUTOSIZE)

        self.get_logger().info('===== Enhanced Perception Node Started =====')

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        cv2.destroyAllWindows()
        super().destroy_node()

    def local_pos_callback(self, msg: PoseStamped):
        """Track current altitude for pixel-to-world conversion."""
        self.current_altitude = msg.pose.position.z

    def image_callback(self, msg: Image):
        """Main processing loop."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        debug_frame = cv_image.copy()
        
        try:
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except Exception as e:
            self.get_logger().error(f'HSV conversion error: {e}')
            return

        # Draw camera center crosshairs
        cv2.circle(debug_frame, (self.cam_center_x, self.cam_center_y), 10, (0, 255, 0), 2)
        cv2.line(debug_frame, (self.cam_center_x - 30, self.cam_center_y), 
                 (self.cam_center_x + 30, self.cam_center_y), (0, 255, 0), 2)
        cv2.line(debug_frame, (self.cam_center_x, self.cam_center_y - 30), 
                 (self.cam_center_x, self.cam_center_y + 30), (0, 255, 0), 2)

        # Run detections
        self.find_zones(cv_image, hsv, debug_frame, msg.header)
        
        if self.model:
            self.find_objects(cv_image, debug_frame, msg.header)

        # Display
        try:
            cv2.imshow(self.debug_window_name, debug_frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'Display error: {e}', throttle_duration_sec=10.0)

        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, 'bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f'Debug publish error: {e}', throttle_duration_sec=10.0)

    def find_zones(self, bgr_image, hsv_image, debug_frame, header: Header):
        """Find white classification zones and blue disaster zones."""
        try:
            # White Zone (BGR)
            mask_white = cv2.inRange(bgr_image, self.white_lower_bgr, self.white_upper_bgr)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, self.morph_kernel)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, self.morph_kernel)

            # Blue Zone (HSV)
            mask_blue = cv2.inRange(hsv_image, self.blue_lower_hsv, self.blue_upper_hsv)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, self.morph_kernel)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, self.morph_kernel)

        except Exception as e:
            self.get_logger().error(f"Mask creation error: {e}")
            return

        # Process White Zone
        try:
            contours_w, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours_w:
                largest_w = max(contours_w, key=cv2.contourArea)
                area_w = cv2.contourArea(largest_w)
                
                if area_w > self.min_zone_area:
                    M = cv2.moments(largest_w)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        self.publish_point(self.white_zone_pub, header, cX, cY)
                        
                        cv2.drawContours(debug_frame, [largest_w], -1, (0, 255, 0), 3)
                        cv2.circle(debug_frame, (cX, cY), 10, (0, 255, 0), -1)
                        cv2.putText(debug_frame, "WHITE ZONE", (cX - 60, cY - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        self.get_logger().info(f"WHITE ZONE @ ({cX},{cY})", 
                                             throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f"White zone error: {e}")

        # Process Blue Zone
        try:
            contours_b, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours_b:
                largest_b = max(contours_b, key=cv2.contourArea)
                area_b = cv2.contourArea(largest_b)
                
                if area_b > self.min_zone_area:
                    M = cv2.moments(largest_b)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        self.publish_point(self.blue_zone_pub, header, cX, cY)
                        
                        cv2.drawContours(debug_frame, [largest_b], -1, (255, 0, 0), 3)
                        cv2.circle(debug_frame, (cX, cY), 10, (255, 0, 0), -1)
                        cv2.putText(debug_frame, "BLUE DISASTER ZONE", (cX - 80, cY - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        # Draw centering line
                        cv2.line(debug_frame, (self.cam_center_x, self.cam_center_y), 
                                (cX, cY), (0, 255, 255), 3)
                        
                        self.get_logger().info(f"BLUE ZONE @ ({cX},{cY})", 
                                             throttle_duration_sec=5.0)

                        # Search for red payload target inside blue zone
                        self.find_payload_target(hsv_image, debug_frame, header, largest_b)
        except Exception as e:
            self.get_logger().error(f"Blue zone error: {e}")

    def find_payload_target(self, hsv_image, debug_frame, header: Header, blue_contour):
        """Find red circular payload target inside blue zone."""
        try:
            # Red mask (HSV wrap-around)
            mask_r1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
            mask_r2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)

            # Mask only inside blue zone
            blue_mask = np.zeros(hsv_image.shape[:2], dtype="uint8")
            cv2.drawContours(blue_mask, [blue_contour], -1, 255, -1)

            mask_final = cv2.bitwise_and(mask_red, mask_red, mask=blue_mask)
            mask_final = cv2.medianBlur(mask_final, 5)

            # Detect circles
            circles = cv2.HoughCircles(mask_final, cv2.HOUGH_GRADIENT, 1, 100,
                                      param1=50, param2=15, minRadius=10, maxRadius=200)
            
            if circles is not None:
                circ = np.uint16(np.around(circles))[0, 0]
                cX, cY, r = int(circ[0]), int(circ[1]), int(circ[2])
                
                # Store for reference
                self.last_payload_target_pixel = (cX, cY)
                
                # Publish pixel coordinates
                pixel_pt = PointStamped()
                pixel_pt.header = header
                pixel_pt.point.x = float(cX)
                pixel_pt.point.y = float(cY)
                pixel_pt.point.z = 0.0
                self.payload_target_pixel_pub.publish(pixel_pt)
                
                # Convert to world position and publish as PoseStamped
                world_pose = self.pixel_to_world_pose(cX, cY, header)
                if world_pose:
                    self.payload_target_pose_pub.publish(world_pose)
                
                # Draw on debug frame
                cv2.circle(debug_frame, (cX, cY), r, (0, 0, 255), 4)
                cv2.circle(debug_frame, (cX, cY), 10, (0, 0, 255), -1)
                
                # Draw error lines
                cv2.line(debug_frame, (self.cam_center_x, self.cam_center_y), 
                        (cX, cY), (0, 255, 255), 3)
                
                # Display pixel errors
                error_x = cX - self.cam_center_x
                error_y = cY - self.cam_center_y
                cv2.putText(debug_frame, f"PAYLOAD TARGET", (cX - 70, cY - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(debug_frame, f"Err X:{error_x} Y:{error_y}", (cX - 70, cY + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                self.get_logger().info(
                    f"PAYLOAD TARGET @ pixel({cX},{cY}) err({error_x},{error_y})", 
                    throttle_duration_sec=2.0)
                    
        except Exception as e:
            self.get_logger().error(f"Payload target error: {e}")

    def pixel_to_world_pose(self, pixel_x, pixel_y, header: Header):
        """
        Convert pixel coordinates to world position using camera model.
        Assumes downward-facing camera and known altitude.
        """
        try:
            if self.current_altitude <= 0.5:
                return None  # Not airborne yet
            
            # Camera model: pixels to angles
            fov_h_rad = np.deg2rad(self.cam_fov_h)
            fov_v_rad = fov_h_rad * (self.cam_height / self.cam_width)
            
            # Pixel offset from center
            dx_pixels = pixel_x - self.cam_center_x
            dy_pixels = pixel_y - self.cam_center_y
            
            # Convert to angles
            angle_x = (dx_pixels / self.cam_width) * fov_h_rad
            angle_y = (dy_pixels / self.cam_height) * fov_v_rad
            
            # Project to ground using altitude (small angle approximation)
            # For downward camera: positive pixel X → positive body Y (right)
            #                     positive pixel Y → negative body X (backward)
            world_offset_y = self.current_altitude * np.tan(angle_x)  # Right/Left
            world_offset_x = -self.current_altitude * np.tan(angle_y)  # Forward/Back
            
            # Create PoseStamped (relative to current drone position)
            # Navigation node will add these offsets to current position
            pose = PoseStamped()
            pose.header = header
            pose.header.frame_id = 'camera_relative'  # Custom frame
            pose.pose.position.x = world_offset_x
            pose.pose.position.y = world_offset_y
            pose.pose.position.z = 0.0  # Keep current altitude
            pose.pose.orientation.w = 1.0
            
            return pose
            
        except Exception as e:
            self.get_logger().error(f"Pixel-to-world conversion error: {e}")
            return None

    def find_objects(self, bgr_image, debug_frame, header: Header):
        """
        Run YOLO detection.
        - Publish detections above confidence threshold
        - Log ALL detections (even below threshold) for YAML
        """
        frame_detections = []  # All objects in this frame
        object_counts = {}
        
        try:
            # Run inference with display threshold
            results = self.model(bgr_image, verbose=False, conf=self.yolo_conf)
            
            # Also run with 0.0 threshold for complete logging
            #results_all = self.model(bgr_image, verbose=False, conf=self.yolo_conf_log)
            
            # Log ALL detections (for YAML)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    
                    detection_record = {
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                    }
                    frame_detections.append(detection_record)
            
            # Publish main detections (above threshold)
            detections_msg = Detection2DArray()
            detections_msg.header = header

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    cX = (x1 + x2) // 2
                    cY = (y1 + y2) // 2
                    
                    # Count for display
                    #object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    
                    # Create Detection2D
                    det = Detection2D()
                    det.header = header
                    det.bbox.center.position.x = float(cX)
                    det.bbox.center.position.y = float(cY)
                    det.bbox.size_x = float(x2 - x1)
                    det.bbox.size_y = float(y2 - y1)
                    
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = class_name
                    hyp.hypothesis.score = conf
                    det.results.append(hyp)
                    detections_msg.detections.append(det)

                    # Draw on debug frame
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(debug_frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.circle(debug_frame, (cX, cY), 5, (255, 0, 255), -1)

            # Publish detections
            if detections_msg.detections:
                self.objects_pub.publish(detections_msg)
                self.get_logger().info(
                    f"DETECTED {len(detections_msg.detections)} objects", 
                    throttle_duration_sec=2.0)
            
            # Publish ALL detections as JSON for logging
            if frame_detections:
                log_msg = String()
                log_msg.data = json.dumps(frame_detections)
                self.all_detections_pub.publish(log_msg)
                self.all_detections_buffer.extend(frame_detections)
            
            # Draw counts
            y_offset = 30
            cv2.putText(debug_frame, "=== Object Counts ===", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            '''for class_name, count in object_counts.items():
                y_offset += 30
                cv2.putText(debug_frame, f"{class_name}: {count}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)'''
        
        except Exception as e:
            self.get_logger().error(f"YOLO error: {e}")

    def publish_point(self, publisher, header: Header, x: int, y: int):
        """Helper to publish PointStamped."""
        try:
            p = PointStamped()
            p.header = header
            p.point.x = float(x)
            p.point.y = float(y)
            p.point.z = 0.0
            publisher.publish(p)
        except Exception as e:
            self.get_logger().error(f"Point publish error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EnhancedPerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Ctrl+C detected, shutting down.")
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Perception node shutdown complete.")


if __name__ == '__main__':
    main()
'''
import rclpy
from rclpy.node import Node
import cv2  # OpenCV
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO # Ensure this is installed (pip install ultralytics)
import os

# Import message types
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

class PerceptionNode(Node):
    """
    This node handles all computer vision tasks for the AeroTHON 2025 mission.
    It subscribes to a raw camera feed and publishes detections for:
    1. White Classification Zones (as a center point) - Uses BGR Thresholding
    2. Blue Disaster Zones (as a center point) - Uses HSV Thresholding
    3. YOLOv8 Objects (as a Detection2DArray)
    
    All detection parameters are tunable via ROS2 parameters for easy on-site calibration.
    It also provides a live, annotated debug window with object counts and a centering reticle.
    """
    def __init__(self):
        super().__init__('perception_node')
        self.get_logger().info(f'===== {self.get_name()} Initializing =====')

        # --- Parameters (Tunable via launch file or command line) ---
        
        # --- 1. Topics ---
        self.declare_parameter('camera_topic', '/camera/camera/image_raw')
        
        # --- 2. File Paths ---
        self.declare_parameter('yolo_model_path', 'final_model_best.pt') # Path to your custom model

        # --- 3. YOLOv8 Config ---
        self.declare_parameter('yolo_conf_threshold', 0.2) # Confidence threshold

        # --- 4. Detection Area Thresholds ---
        self.declare_parameter('min_zone_area', 2000.0) # Min contour area to be a valid zone

        # --- 5. Camera Geometry ---
        self.declare_parameter('camera_center_x_px', 320) # X-pixel coord of camera center
        self.declare_parameter('camera_center_y_px', 240) # Y-pixel coord of camera center

        # --- 6. HSV/BGR Thresholds (CRITICAL: TUNE THESE ON-SITE) ---
        self.declare_parameter('white_lower_bgr', [190, 190, 190])
        self.declare_parameter('white_upper_bgr', [255, 255, 255])
        self.declare_parameter('blue_lower_hsv', [90, 80, 80])
        self.declare_parameter('blue_upper_hsv', [135, 255, 255])
        self.declare_parameter('red_lower1_hsv', [0, 100, 80])
        self.declare_parameter('red_upper1_hsv', [10, 255, 255])
        self.declare_parameter('red_lower2_hsv', [165, 100, 80])
        self.declare_parameter('red_upper2_hsv', [180, 255, 255])

        # --- Get Parameter Values ---
        camera_topic = self.get_parameter('camera_topic').value
        model_path = self.get_parameter('yolo_model_path').value
        self.yolo_conf = self.get_parameter('yolo_conf_threshold').value
        self.min_zone_area = self.get_parameter('min_zone_area').value
        self.cam_center_x = self.get_parameter('camera_center_x_px').value
        self.cam_center_y = self.get_parameter('camera_center_y_px').value

        # --- Load Thresholds as NumPy arrays ---
        self.white_lower_bgr = np.array(self.get_parameter('white_lower_bgr').value)
        self.white_upper_bgr = np.array(self.get_parameter('white_upper_bgr').value)
        self.blue_lower_hsv = np.array(self.get_parameter('blue_lower_hsv').value)
        self.blue_upper_hsv = np.array(self.get_parameter('blue_upper_hsv').value)
        self.red_lower1 = np.array(self.get_parameter('red_lower1_hsv').value)
        self.red_upper1 = np.array(self.get_parameter('red_upper1_hsv').value)
        self.red_lower2 = np.array(self.get_parameter('red_lower2_hsv').value)
        self.red_upper2 = np.array(self.get_parameter('red_upper2_hsv').value)

        # Kernel for morphological operations (noise reduction)
        self.morph_kernel = np.ones((5, 5), np.uint8)
        
        # --- CV Bridge ---
        self.bridge = CvBridge()

        # --- YOLOv8 Model ---
        self.model = None
        if not os.path.exists(model_path):
            self.get_logger().error(f"YOLO model NOT FOUND at: {os.path.abspath(model_path)}")
            self.get_logger().error("YOLO detection will be disabled.")
        else:
            try:
                self.model = YOLO(model_path)
                self.get_logger().info(f'YOLO model loaded successfully from: {model_path}')
            except Exception as e:
                self.get_logger().error(f'YOLO model loading FAILED: {e}')

        # --- Subscribers ---
        self.image_sub = self.create_subscription(
            Image, 
            camera_topic, 
            self.image_callback, 
            10) # QoS depth 10
        self.get_logger().info(f'Subscribed to camera topic: {camera_topic}')

        # --- Publishers (Topics match your navigation node) ---
        self.white_zone_pub = self.create_publisher(PointStamped, '/detection/white_zone_center', 10)
        self.blue_zone_pub = self.create_publisher(PointStamped, '/detection/blue_zone_center', 10)
        self.objects_pub = self.create_publisher(Detection2DArray, '/detection/objects', 10)
        
        # --- Publishers (Optional / Debug) ---
        self.target_pub = self.create_publisher(PointStamped, '/detection/payload_target_center', 10)
        self.debug_image_pub = self.create_publisher(Image, '/detection/debug_image', 10)
        
        # --- Create OpenCV window for the debug feed ---
        self.debug_window_name = "AeroTHON Perception Debug"
        cv2.namedWindow(self.debug_window_name, cv2.WINDOW_AUTOSIZE)
        self.get_logger().info(f"Created debug window: '{self.debug_window_name}'")

        self.get_logger().info(f'===== {self.get_name()} Started =====')

    def destroy_node(self):
        """Clean up resources on shutdown."""
        self.get_logger().info("Shutting down, closing debug window.")
        cv2.destroyAllWindows()
        super().destroy_node()

    def image_callback(self, msg: Image):
        """Main processing loop. Runs on every new image."""
        try:
            # Convert ROS Image message to OpenCV image (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # Create a copy for drawing annotations
        debug_frame = cv_image.copy()
        
        try:
            # Convert to HSV color space *once* for blue/red detection
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except Exception as e:
            self.get_logger().error(f'Error converting to HSV: {e}')
            return

        # --- Draw Camera Center Crosshairs (for centering) ---
        try:
            cv2.circle(debug_frame, (self.cam_center_x, self.cam_center_y), 7, (0, 255, 0), 2)
            cv2.line(debug_frame, (self.cam_center_x, self.cam_center_y - 20), (self.cam_center_x, self.cam_center_y + 20), (0, 255, 0), 2)
            cv2.line(debug_frame, (self.cam_center_x - 20, self.cam_center_y), (self.cam_center_x + 20, self.cam_center_y), (0, 255, 0), 2)
        except Exception as e:
             self.get_logger().warn(f'Error drawing reticle: {e}', throttle_duration_sec=10.0)

        # --- Run All Detection Functions ---
        # 1. Find White (BGR) and Blue (HSV) zones
        self.find_zones(cv_image, hsv, debug_frame, msg.header)
        
        # 2. Find YOLO objects (runs on BGR image)
        if self.model:
            self.find_objects(cv_image, debug_frame, msg.header)

        # --- Show image in the OpenCV debug window ---
        try:
            cv2.imshow(self.debug_window_name, debug_frame)
            cv2.waitKey(1) # This 1ms wait is required to update the window
        except Exception as e:
            # This can fail if running headless (e.g., on the drone)
            self.get_logger().warn(f'cv2.imshow Error: {e}', throttle_duration_sec=10.0)

        # --- Publish Debug Image (for Rviz) ---
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, 'bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f'Debug publishing Error: {e}', throttle_duration_sec=10.0)

    def find_zones(self, bgr_image, hsv_image, debug_frame, header: Header):
        """
        Finds white and blue zones.
        - White detection uses BGR image.
        - Blue detection uses HSV image.
        """
        try:
            # --- White Zone (BGR Thresholding) ---
            mask_white = cv2.inRange(bgr_image, self.white_lower_bgr, self.white_upper_bgr)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, self.morph_kernel)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, self.morph_kernel)

            # --- Blue Zone (HSV Thresholding) ---
            mask_blue = cv2.inRange(hsv_image, self.blue_lower_hsv, self.blue_upper_hsv)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, self.morph_kernel)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, self.morph_kernel)

        except Exception as e:
            self.get_logger().error(f"Mask creation error: {e}")
            return

        # --- Process White Zone Contours ---
        try:
            contours_w, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours_w:
                largest_w = max(contours_w, key=cv2.contourArea)
                area_w = cv2.contourArea(largest_w)
                
                if area_w > self.min_zone_area:
                    M = cv2.moments(largest_w)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        self.publish_point(self.white_zone_pub, header, cX, cY)
                        
                        cv2.drawContours(debug_frame, [largest_w], -1, (0, 255, 0), 3) # Green
                        cv2.circle(debug_frame, (cX, cY), 7, (0, 255, 0), -1)
                        cv2.putText(debug_frame, "WHITE ZONE", (cX - 20, cY - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        self.get_logger().info(f"** DETECTED: White Zone @ ({cX},{cY}) **", throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f"White zone processing error: {e}")

        # --- Process Blue Zone Contours ---
        try:
            contours_b, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours_b:
                largest_b = max(contours_b, key=cv2.contourArea)
                area_b = cv2.contourArea(largest_b)
                
                if area_b > self.min_zone_area:
                    M = cv2.moments(largest_b)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        self.publish_point(self.blue_zone_pub, header, cX, cY)
                        
                        cv2.drawContours(debug_frame, [largest_b], -1, (255, 0, 0), 3) # Blue
                        cv2.circle(debug_frame, (cX, cY), 7, (255, 0, 0), -1)
                        cv2.putText(debug_frame, "BLUE ZONE (DISASTER)", (cX - 20, cY - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # --- NEW: Draw Centering Error Line ---
                        cv2.line(debug_frame, (self.cam_center_x, self.cam_center_y), (cX, cY), (0, 255, 255), 2) # Yellow line
                        
                        self.get_logger().info(f"** DETECTED: Blue Zone @ ({cX},{cY}) **", throttle_duration_sec=5.0)

                        self.find_payload_target(hsv_image, debug_frame, header, largest_b)
        except Exception as e:
            self.get_logger().error(f"Blue zone processing error: {e}")

    def find_payload_target(self, hsv_image, debug_frame, header: Header, blue_contour):
        """Finds a red circular target *only* within the bounds of the blue zone."""
        try:
            mask_r1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
            mask_r2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
            mask_red_combined = cv2.bitwise_or(mask_r1, mask_r2)

            blue_mask = np.zeros(hsv_image.shape[:2], dtype="uint8")
            cv2.drawContours(blue_mask, [blue_contour], -1, 255, -1) 

            mask_final = cv2.bitwise_and(mask_red_combined, mask_red_combined, mask=blue_mask)
            mask_final = cv2.medianBlur(mask_final, 5) 

            circles = cv2.HoughCircles(mask_final, cv2.HOUGH_GRADIENT, 1, 100,
                                         param1=50, param2=15, minRadius=10, maxRadius=150)
            
            if circles is not None:
                circ = np.uint16(np.around(circles))[0, 0]
                cX, cY, r = circ[0], circ[1], circ[2]
                
                self.publish_point(self.target_pub, header, cX, cY)
                cv2.circle(debug_frame, (cX, cY), r, (0, 0, 255), 3) # Red
                cv2.circle(debug_frame, (cX, cY), 7, (0, 0, 255), -1)
                cv2.putText(debug_frame, "PAYLOAD TARGET", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                self.get_logger().info(f"** DETECTED: Payload Target @ ({cX},{cY}) **", throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f"Payload target finding error: {e}")

    def find_objects(self, bgr_image, debug_frame, header: Header):
        """Runs YOLOv8 object detection, publishes a Detection2DArray, and draws counts on debug frame."""
        
        # --- NEW: Dictionary to hold counts for this frame ---
        object_counts = {}

        try:
            results = self.model(bgr_image, verbose=False, conf=self.yolo_conf)
            
            detections_msg = Detection2DArray()
            detections_msg.header = header

            for r in results:
                for box in r.boxes:
                    cls_id_int = int(box.cls[0])
                    class_name = self.model.names[cls_id_int]
                    conf = float(box.conf[0])
                    
                    # --- NEW: Increment count for this class ---
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    cX = (x1 + x2) // 2
                    cY = (y1 + y2) // 2
                    
                    # --- Create Detection2D message ---
                    det = Detection2D()
                    det.header = header
                    det.bbox.center.position.x = float(cX)
                    det.bbox.center.position.y = float(cY)
                    det.bbox.size_x = float(x2 - x1)
                    det.bbox.size_y = float(y2 - y1)
                    
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = class_name 
                    hyp.hypothesis.score = conf
                    det.results.append(hyp)
                    
                    detections_msg.detections.append(det)

                    # --- Draw on debug frame ---
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow
                    cv2.putText(debug_frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.circle(debug_frame, (cX, cY), 5, (255, 0, 255), -1) # Magenta dot

            if detections_msg.detections:
                self.objects_pub.publish(detections_msg)
                self.get_logger().info(f"** DETECTED: {len(detections_msg.detections)} objects **", throttle_duration_sec=2.0)
            
            # --- NEW: Draw counts on the debug frame ---
            y_offset = 30
            cv2.putText(debug_frame, "--- Frame Counts ---", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            for class_name, count in object_counts.items():
                y_offset += 25
                text = f"{class_name}: {count}"
                cv2.putText(debug_frame, text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        except Exception as e:
            self.get_logger().error(f"YOLO inference error: {e}")

    def publish_point(self, publisher: Node.create_publisher, header: Header, x: int, y: int):
        """Helper function to create and publish a PointStamped message."""
        try:
            p = PointStamped()
            p.header = header
            p.point.x = float(x)
            p.point.y = float(y)
            p.point.z = 0.0  # 2D detection, Z is 0
            publisher.publish(p)
        except Exception as e:
            self.get_logger().error(f"Point publishing error: {e}")

# --- Main ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Ctrl+C detected, shutting down perception node.")
    except Exception as e:
        print(f"Exception in perception node: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Perception node shutdown complete.")

if __name__ == '__main__':
    main()
---------------------
import rclpy
from rclpy.node import Node
import cv2  # OpenCV
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO # Ensure this is installed (pip install ultralytics)
import os

# Import message types
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

class PerceptionNode(Node):
    """
    This node handles all computer vision tasks for the AeroTHON 2025 mission.
    It subscribes to a raw camera feed and publishes detections for:
    1. White Classification Zones (as a center point) - Uses BGR Thresholding
    2. Blue Disaster Zones (as a center point) - Uses HSV Thresholding
    3. YOLOv8 Objects (as a Detection2DArray)
    
    All detection parameters are tunable via ROS2 parameters for easy on-site calibration.
    It also provides a live, annotated debug window using OpenCV.
    """
    def __init__(self):
        super().__init__('perception_node')
        self.get_logger().info(f'===== {self.get_name()} Initializing =====')

        # --- Parameters (Tunable via launch file or command line) ---
        
        # --- 1. Topics ---
        self.declare_parameter('camera_topic', '/camera/camera/image_raw')
        
        # --- 2. File Paths ---
        self.declare_parameter('yolo_model_path', 'final_model_best.pt') # Path to your custom model

        # --- 3. YOLOv8 Config ---
        self.declare_parameter('yolo_conf_threshold', 0.1) # Confidence threshold

        # --- 4. Detection Area Thresholds ---
        self.declare_parameter('min_zone_area', 2000.0) # Min contour area to be a valid zone

        # --- 5. HSV/BGR Thresholds (CRITICAL: TUNE THESE ON-SITE) ---
        
        # ** NEW WHITE LOGIC (BGR) **
        # Detects white by looking for high values in Blue, Green, AND Red channels.
        # This is much more robust to outdoor lighting than using HSV's Saturation/Value.
        self.declare_parameter('white_lower_bgr', [190, 190, 190])
        self.declare_parameter('white_upper_bgr', [255, 255, 255])
        
        # Blue (HSV)
        self.declare_parameter('blue_lower_hsv', [90, 80, 80])
        self.declare_parameter('blue_upper_hsv', [135, 255, 255])
        
        # Red (HSV) - for payload target (handles HSV wrap-around)
        self.declare_parameter('red_lower1_hsv', [0, 100, 80])
        self.declare_parameter('red_upper1_hsv', [10, 255, 255])
        self.declare_parameter('red_lower2_hsv', [165, 100, 80])
        self.declare_parameter('red_upper2_hsv', [180, 255, 255])

        # --- Get Parameter Values ---
        camera_topic = self.get_parameter('camera_topic').value
        model_path = self.get_parameter('yolo_model_path').value
        self.yolo_conf = self.get_parameter('yolo_conf_threshold').value
        self.min_zone_area = self.get_parameter('min_zone_area').value

        # --- Load Thresholds as NumPy arrays ---
        self.white_lower_bgr = np.array(self.get_parameter('white_lower_bgr').value)
        self.white_upper_bgr = np.array(self.get_parameter('white_upper_bgr').value)
        
        self.blue_lower_hsv = np.array(self.get_parameter('blue_lower_hsv').value)
        self.blue_upper_hsv = np.array(self.get_parameter('blue_upper_hsv').value)
        
        self.red_lower1 = np.array(self.get_parameter('red_lower1_hsv').value)
        self.red_upper1 = np.array(self.get_parameter('red_upper1_hsv').value)
        self.red_lower2 = np.array(self.get_parameter('red_lower2_hsv').value)
        self.red_upper2 = np.array(self.get_parameter('red_upper2_hsv').value)

        # Kernel for morphological operations (noise reduction)
        self.morph_kernel = np.ones((5, 5), np.uint8)
        
        # --- CV Bridge ---
        self.bridge = CvBridge()

        # --- YOLOv8 Model ---
        self.model = None
        if not os.path.exists(model_path):
            self.get_logger().error(f"YOLO model NOT FOUND at: {os.path.abspath(model_path)}")
            self.get_logger().error("YOLO detection will be disabled.")
        else:
            try:
                self.model = YOLO(model_path)
                self.get_logger().info(f'YOLO model loaded successfully from: {model_path}')
            except Exception as e:
                self.get_logger().error(f'YOLO model loading FAILED: {e}')

        # --- Subscribers ---
        self.image_sub = self.create_subscription(
            Image, 
            camera_topic, 
            self.image_callback, 
            10) # QoS depth 10
        self.get_logger().info(f'Subscribed to camera topic: {camera_topic}')

        # --- Publishers (Topics match your navigation node) ---
        self.white_zone_pub = self.create_publisher(PointStamped, '/detection/white_zone_center', 10)
        self.blue_zone_pub = self.create_publisher(PointStamped, '/detection/blue_zone_center', 10)
        self.objects_pub = self.create_publisher(Detection2DArray, '/detection/objects', 10)
        
        # --- Publishers (Optional / Debug) ---
        self.target_pub = self.create_publisher(PointStamped, '/detection/payload_target_center', 10)
        self.debug_image_pub = self.create_publisher(Image, '/detection/debug_image', 10)
        
        # --- Create OpenCV window for the debug feed ---
        self.debug_window_name = "AeroTHON Perception Debug"
        cv2.namedWindow(self.debug_window_name, cv2.WINDOW_AUTOSIZE)
        self.get_logger().info(f"Created debug window: '{self.debug_window_name}'")

        self.get_logger().info(f'===== {self.get_name()} Started =====')

    def destroy_node(self):
        """Clean up resources on shutdown."""
        self.get_logger().info("Shutting down, closing debug window.")
        cv2.destroyAllWindows()
        super().destroy_node()

    def image_callback(self, msg: Image):
        """Main processing loop. Runs on every new image."""
        try:
            # Convert ROS Image message to OpenCV image (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # Create a copy for drawing annotations
        debug_frame = cv_image.copy()
        
        try:
            # Convert to HSV color space *once* for blue/red detection
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except Exception as e:
            self.get_logger().error(f'Error converting to HSV: {e}')
            return

        # --- Run All Detection Functions ---
        # 1. Find White (BGR) and Blue (HSV) zones
        self.find_zones(cv_image, hsv, debug_frame, msg.header)
        
        # 2. Find YOLO objects (runs on BGR image)
        if self.model:
            self.find_objects(cv_image, debug_frame, msg.header)

        # --- Show image in the OpenCV debug window ---
        try:
            cv2.imshow(self.debug_window_name, debug_frame)
            cv2.waitKey(1) # This 1ms wait is required to update the window
        except Exception as e:
            # This can fail if running headless (e.g., on the drone)
            self.get_logger().warn(f'cv2.imshow Error: {e}', throttle_duration_sec=10.0)

        # --- Publish Debug Image (for Rviz) ---
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, 'bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f'Debug publishing Error: {e}', throttle_duration_sec=10.0)

    def find_zones(self, bgr_image, hsv_image, debug_frame, header: Header):
        """
        Finds white and blue zones.
        - White detection uses BGR image.
        - Blue detection uses HSV image.
        """
        try:
            # --- White Zone (BGR Thresholding) ---
            # This is more robust to shadows/sunlight than HSV's S and V channels
            mask_white = cv2.inRange(bgr_image, self.white_lower_bgr, self.white_upper_bgr)
            # Noise reduction
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, self.morph_kernel)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, self.morph_kernel)

            # --- Blue Zone (HSV Thresholding) ---
            mask_blue = cv2.inRange(hsv_image, self.blue_lower_hsv, self.blue_upper_hsv)
            # Noise reduction
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, self.morph_kernel)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, self.morph_kernel)

        except Exception as e:
            self.get_logger().error(f"Mask creation error: {e}")
            return

        # --- Process White Zone Contours ---
        try:
            contours_w, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours_w:
                largest_w = max(contours_w, key=cv2.contourArea)
                area_w = cv2.contourArea(largest_w)
                
                if area_w > self.min_zone_area:
                    M = cv2.moments(largest_w)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # Publish the center point for mission_node
                        self.publish_point(self.white_zone_pub, header, cX, cY)
                        
                        # Draw on debug frame
                        cv2.drawContours(debug_frame, [largest_w], -1, (0, 255, 0), 3) # Green
                        cv2.circle(debug_frame, (cX, cY), 7, (0, 255, 0), -1)
                        cv2.putText(debug_frame, "WHITE ZONE", (cX - 20, cY - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Throttle log to avoid spam
                        self.get_logger().info(f"** DETECTED: White Zone @ ({cX},{cY}) **", throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f"White zone processing error: {e}")

        # --- Process Blue Zone Contours ---
        try:
            contours_b, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours_b:
                largest_b = max(contours_b, key=cv2.contourArea)
                area_b = cv2.contourArea(largest_b)
                
                if area_b > self.min_zone_area:
                    M = cv2.moments(largest_b)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # Publish the center point for mission_node
                        self.publish_point(self.blue_zone_pub, header, cX, cY)
                        
                        # Draw on debug frame
                        cv2.drawContours(debug_frame, [largest_b], -1, (255, 0, 0), 3) # Blue
                        cv2.circle(debug_frame, (cX, cY), 7, (255, 0, 0), -1)
                        cv2.putText(debug_frame, "BLUE ZONE (DISASTER)", (cX - 20, cY - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        self.get_logger().info(f"** DETECTED: Blue Zone @ ({cX},{cY}) **", throttle_duration_sec=5.0)

                        # --- If blue zone found, search for red target *inside* it ---
                        self.find_payload_target(hsv_image, debug_frame, header, largest_b)
        except Exception as e:
            self.get_logger().error(f"Blue zone processing error: {e}")

    def find_payload_target(self, hsv_image, debug_frame, header: Header, blue_contour):
        """Finds a red circular target *only* within the bounds of the blue zone."""
        try:
            # Create red mask (handling wrap-around)
            mask_r1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
            mask_r2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
            mask_red_combined = cv2.bitwise_or(mask_r1, mask_r2)

            # Create a mask *only* for the area inside the blue contour
            blue_mask = np.zeros(hsv_image.shape[:2], dtype="uint8")
            cv2.drawContours(blue_mask, [blue_contour], -1, 255, -1) # -1 thickness fills the contour

            # Final mask: red pixels that are ALSO inside the blue zone
            mask_final = cv2.bitwise_and(mask_red_combined, mask_red_combined, mask=blue_mask)
            mask_final = cv2.medianBlur(mask_final, 5) # Blur for HoughCircles

            # Find circles
            circles = cv2.HoughCircles(mask_final, cv2.HOUGH_GRADIENT, 1, 100,
                                         param1=50, param2=15, minRadius=10, maxRadius=150)
            
            if circles is not None:
                # Get the first (and likely only) circle
                circ = np.uint16(np.around(circles))[0, 0]
                cX, cY, r = circ[0], circ[1], circ[2]
                
                self.publish_point(self.target_pub, header, cX, cY)
                cv2.circle(debug_frame, (cX, cY), r, (0, 0, 255), 3) # Red
                cv2.circle(debug_frame, (cX, cY), 7, (0, 0, 255), -1)
                cv2.putText(debug_frame, "PAYLOAD TARGET", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                self.get_logger().info(f"** DETECTED: Payload Target @ ({cX},{cY}) **", throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f"Payload target finding error: {e}")

    def find_objects(self, bgr_image, debug_frame, header: Header):
        """Runs YOLOv8 object detection and publishes a Detection2DArray."""
        try:
            # Run YOLO inference
            results = self.model(bgr_image, verbose=False, conf=self.yolo_conf)
            
            detections_msg = Detection2DArray()
            detections_msg.header = header

            for r in results:
                for box in r.boxes:
                    # Get class ID and name
                    cls_id_int = int(box.cls[0])
                    class_name = self.model.names[cls_id_int]
                    conf = float(box.conf[0])
                    
                    # Get BBox coordinates
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    cX = (x1 + x2) // 2
                    cY = (y1 + y2) // 2
                    
                    # --- Create Detection2D message ---
                    det = Detection2D()
                    det.header = header
                    det.bbox.center.position.x = float(cX)
                    det.bbox.center.position.y = float(cY)
                    det.bbox.size_x = float(x2 - x1)
                    det.bbox.size_y = float(y2 - y1)
                    
                    hyp = ObjectHypothesisWithPose()
                    # Per vision_msgs, class_id is a string.
                    # Your mission_node correctly uses this string as a dict key.
                    hyp.hypothesis.class_id = class_name 
                    hyp.hypothesis.score = conf
                    det.results.append(hyp)
                    
                    detections_msg.detections.append(det)

                    # --- Draw on debug frame ---
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow
                    cv2.putText(debug_frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.circle(debug_frame, (cX, cY), 5, (255, 0, 255), -1) # Magenta dot

            # Publish the array if any objects were found
            if detections_msg.detections:
                self.objects_pub.publish(detections_msg)
                self.get_logger().info(f"** DETECTED: {len(detections_msg.detections)} objects **", throttle_duration_sec=2.0)
        
        except Exception as e:
            self.get_logger().error(f"YOLO inference error: {e}")

    def publish_point(self, publisher: Node.create_publisher, header: Header, x: int, y: int):
        """Helper function to create and publish a PointStamped message."""
        try:
            p = PointStamped()
            p.header = header
            p.point.x = float(x)
            p.point.y = float(y)
            p.point.z = 0.0  # 2D detection, Z is 0
            publisher.publish(p)
        except Exception as e:
            self.get_logger().error(f"Point publishing error: {e}")

# --- Main ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Ctrl+C detected, shutting down perception node.")
    except Exception as e:
        print(f"Exception in perception node: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Perception node shutdown complete.")

if __name__ == '__main__':
    main()'''