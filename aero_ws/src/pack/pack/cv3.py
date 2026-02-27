#!/usr/bin/env python3
"""
ULTRA-PRECISE Perception Node for AeroTHON 2025
- Optimized for ultra-precise centering
- Better zone detection with improved filtering
- Accurate pixel coordinate publishing
- Robust target detection
"""
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import time
import json

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String


class UltraPrecisePerceptionNode(Node):
    def __init__(self):
        super().__init__('ultra_precise_perception_node')
        self.get_logger().info('===== ULTRA-PRECISE PERCEPTION NODE INITIALIZING =====')

        # Parameters
        self.declare_parameter('camera_topic', '/camera/camera/image_raw')
        self.declare_parameter('yolo_model_path', 'final_model_best.pt')
        self.declare_parameter('yolo_conf_threshold', 0.15)
        self.declare_parameter('min_zone_area', 1000.0)  # Lower for better detection

        # Camera settings
        self.declare_parameter('camera_width', 1024)
        self.declare_parameter('camera_height', 768)
        self.declare_parameter('camera_fov_horizontal_deg', 62.2)

        # OPTIMIZED color thresholds for precise detection
        self.declare_parameter('white_lower_bgr', [180, 180, 180])  # More sensitive
        self.declare_parameter('white_upper_bgr', [255, 255, 255])
        self.declare_parameter('blue_lower_hsv', [95, 120, 40])     # Better blue range
        self.declare_parameter('blue_upper_hsv', [125, 255, 255])
        self.declare_parameter('red_lower1_hsv', [0, 120, 50])      # Better red detection
        self.declare_parameter('red_upper1_hsv', [8, 255, 255])
        self.declare_parameter('red_lower2_hsv', [170, 120, 50])
        self.declare_parameter('red_upper2_hsv', [180, 255, 255])

        # Get parameters
        camera_topic = self.get_parameter('camera_topic').value
        model_path = self.get_parameter('yolo_model_path').value
        self.yolo_conf = self.get_parameter('yolo_conf_threshold').value
        self.min_zone_area = self.get_parameter('min_zone_area').value

        self.cam_width = self.get_parameter('camera_width').value
        self.cam_height = self.get_parameter('camera_height').value
        self.cam_center_x = self.cam_width // 2
        self.cam_center_y = self.cam_height // 2
        self.cam_fov_h = self.get_parameter('camera_fov_horizontal_deg').value

        # Load optimized thresholds
        self.white_lower_bgr = np.array(self.get_parameter('white_lower_bgr').value)
        self.white_upper_bgr = np.array(self.get_parameter('white_upper_bgr').value)
        self.blue_lower_hsv = np.array(self.get_parameter('blue_lower_hsv').value)
        self.blue_upper_hsv = np.array(self.get_parameter('blue_upper_hsv').value)
        self.red_lower1 = np.array(self.get_parameter('red_lower1_hsv').value)
        self.red_upper1 = np.array(self.get_parameter('red_upper1_hsv').value)
        self.red_lower2 = np.array(self.get_parameter('red_lower2_hsv').value)
        self.red_upper2 = np.array(self.get_parameter('red_upper2_hsv').value)

        # Improved morphological kernels
        self.morph_open_kernel = np.ones((3, 3), np.uint8)
        self.morph_close_kernel = np.ones((9, 9), np.uint8)  # Larger for better filling
        
        self.bridge = CvBridge()

        # YOLO Model
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.get_logger().info(f'YOLO model loaded: {model_path}')
            except Exception as e:
                self.get_logger().error(f'YOLO loading failed: {e}')
        else:
            self.get_logger().warning(f'YOLO model not found: {model_path}')

        # Subscribers
        self.image_sub = self.create_subscription(Image, camera_topic, self.image_callback, 5)
        self.local_pos_sub = self.create_subscription(
            PoseStamped, 'mavros/local_position/pose', self.local_pos_callback, 10)

        # Publishers
        self.white_zone_pub = self.create_publisher(PointStamped, '/detection/white_zone_center', 10)
        self.blue_zone_pub = self.create_publisher(PointStamped, '/detection/blue_zone_center', 10)
        self.payload_target_pixel_pub = self.create_publisher(PointStamped, '/detection/payload_target_pixel', 10)
        self.all_detections_pub = self.create_publisher(String, '/detection/all_objects_log', 10)
        self.debug_image_pub = self.create_publisher(Image, '/detection/debug_image', 10)

        # State tracking
        self.current_altitude = 0.0
        self.all_detections_buffer = []

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()

        self.get_logger().info('===== ULTRA-PRECISE PERCEPTION NODE READY =====')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

    def local_pos_callback(self, msg: PoseStamped):
        """Track current altitude"""
        self.current_altitude = msg.pose.position.z

    def image_callback(self, msg: Image):
        """Main processing loop optimized for centering"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        # Update frame dimensions
        h, w, _ = cv_image.shape
        self.cam_width = w
        self.cam_height = h
        self.cam_center_x = w // 2
        self.cam_center_y = h // 2

        debug_frame = cv_image.copy()

        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            self.get_logger().info(f'Processing FPS: {fps:.1f}', throttle_duration_sec=5.0)

        try:
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except Exception as e:
            self.get_logger().error(f'HSV conversion error: {e}')
            return

        # Draw precise camera center with crosshairs
        cv2.circle(debug_frame, (self.cam_center_x, self.cam_center_y), 6, (0, 255, 0), 2)
        cv2.line(debug_frame, (self.cam_center_x - 15, self.cam_center_y),
                 (self.cam_center_x + 15, self.cam_center_y), (0, 255, 0), 2)
        cv2.line(debug_frame, (self.cam_center_x, self.cam_center_y - 15),
                 (self.cam_center_x, self.cam_center_y + 15), (0, 255, 0), 2)

        # Run detections
        self.find_zones(cv_image, hsv, debug_frame, msg.header)

        if self.model:
            self.find_objects(cv_image, debug_frame, msg.header)

        # Display
        try:
            cv2.imshow("Ultra-Precise Perception", debug_frame)
            cv2.waitKey(1)
        except:
            pass

        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, 'bgr8')
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f'Debug image publish error: {e}')

    def find_zones(self, bgr_image, hsv_image, debug_frame, header: Header):
        """Optimized zone detection for precise centering"""
        try:
            # White Zone detection
            mask_white = cv2.inRange(bgr_image, self.white_lower_bgr, self.white_upper_bgr)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, self.morph_open_kernel)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, self.morph_close_kernel)

            # Blue Zone detection
            mask_blue = cv2.inRange(hsv_image, self.blue_lower_hsv, self.blue_upper_hsv)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, self.morph_open_kernel)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, self.morph_close_kernel)

        except Exception as e:
            self.get_logger().error(f"Mask creation error: {e}")
            return

        # Process White Zone
        white_center = self.process_zone_contours(mask_white, debug_frame, header, 
                                                self.white_zone_pub, "WHITE ZONE", (0, 255, 0))

        # Process Blue Zone
        blue_center = self.process_zone_contours(mask_blue, debug_frame, header,
                                               self.blue_zone_pub, "BLUE ZONE", (255, 0, 0))

        # If blue zone found, look for payload target
        if blue_center is not None:
            blue_contour = self.get_largest_contour(mask_blue)
            if blue_contour is not None:
                self.find_payload_target(hsv_image, debug_frame, header, blue_contour, blue_center)

    def process_zone_contours(self, mask, debug_frame, header, publisher, label, color):
        """Process contours and return center coordinates"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > self.min_zone_area:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Publish detection
                    self.publish_point(publisher, header, cX, cY)

                    # Draw on debug frame
                    cv2.drawContours(debug_frame, [largest], -1, color, 2)
                    cv2.circle(debug_frame, (cX, cY), 6, color, -1)
                    
                    # Draw precise centering info
                    error_x = cX - self.cam_center_x
                    error_y = cY - self.cam_center_y
                    cv2.putText(debug_frame, f"{label}", (cX - 40, cY - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(debug_frame, f"Err: ({error_x},{error_y})", (cX - 40, cY + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Draw centering line
                    cv2.line(debug_frame, (self.cam_center_x, self.cam_center_y),
                            (cX, cY), (0, 255, 255), 2)

                    self.get_logger().info(f"{label} center @ ({cX},{cY}) error({error_x},{error_y})", 
                                         throttle_duration_sec=2.0)
                    return (cX, cY)

        except Exception as e:
            self.get_logger().error(f"Zone processing error: {e}")
        
        return None

    def get_largest_contour(self, mask):
        """Get largest contour from mask"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                return max(contours, key=cv2.contourArea)
        except:
            pass
        return None

    def find_payload_target(self, hsv_image, debug_frame, header: Header, blue_contour, blue_center):
        """Find red payload target inside blue zone"""
        try:
            # Red mask
            mask_r1 = cv2.inRange(hsv_image, self.red_lower1, self.red_upper1)
            mask_r2 = cv2.inRange(hsv_image, self.red_lower2, self.red_upper2)
            mask_red = cv2.bitwise_or(mask_r1, mask_r2)

            # Mask only inside blue zone
            blue_mask = np.zeros(hsv_image.shape[:2], dtype="uint8")
            cv2.drawContours(blue_mask, [blue_contour], -1, 255, -1)
            mask_final = cv2.bitwise_and(mask_red, mask_red, mask=blue_mask)
            
            # Improved filtering
            mask_final = cv2.medianBlur(mask_final, 5)
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, self.morph_close_kernel)

            # Detect circles with precise parameters
            circles = cv2.HoughCircles(mask_final, cv2.HOUGH_GRADIENT, dp=1.1, 
                                     minDist=30, param1=50, param2=20,
                                     minRadius=10, maxRadius=100)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Take the circle closest to blue zone center
                best_circle = min(circles[0], key=lambda x: np.sqrt((x[0]-blue_center[0])**2 + (x[1]-blue_center[1])**2))
                cX, cY, r = best_circle

                # Publish pixel coordinates
                pixel_msg = PointStamped()
                pixel_msg.header = header
                pixel_msg.point.x = float(cX)
                pixel_msg.point.y = float(cY)
                pixel_msg.point.z = float(r)
                self.payload_target_pixel_pub.publish(pixel_msg)

                # Draw precise detection
                cv2.circle(debug_frame, (cX, cY), r, (0, 0, 255), 2)
                cv2.circle(debug_frame, (cX, cY), 4, (0, 0, 255), -1)
                
                # Draw centering line to payload
                cv2.line(debug_frame, (self.cam_center_x, self.cam_center_y),
                        (cX, cY), (0, 255, 255), 2)

                # Display precise errors
                error_x = cX - self.cam_center_x
                error_y = cY - self.cam_center_y
                cv2.putText(debug_frame, f"PAYLOAD TARGET", (cX - 50, cY - r - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(debug_frame, f"Err: ({error_x},{error_y})", (cX - 40, cY - r + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                self.get_logger().info(f"Payload target @ ({cX},{cY}) error({error_x},{error_y})", 
                                     throttle_duration_sec=1.0)

        except Exception as e:
            self.get_logger().error(f"Payload target error: {e}")

    def find_objects(self, bgr_image, debug_frame, header: Header):
        """Run YOLO object detection"""
        frame_detections = []

        try:
            # Run inference
            results = self.model(bgr_image, verbose=False, conf=self.yolo_conf)

            # Process results
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

                    # Draw on debug frame
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    cv2.putText(debug_frame, f"{class_name} {conf:.2f}", 
                               (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Publish detections for logging
            if frame_detections:
                try:
                    log_msg = String()
                    log_msg.data = json.dumps(frame_detections)
                    self.all_detections_pub.publish(log_msg)
                    self.all_detections_buffer.extend(frame_detections)
                except Exception as e:
                    self.get_logger().error(f"Detection log error: {e}")

            # Show detection count
            cv2.putText(debug_frame, f"Objects: {len(frame_detections)}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        except Exception as e:
            self.get_logger().error(f"YOLO detection error: {e}")

    def publish_point(self, publisher, header: Header, x: int, y: int):
        """Helper to publish PointStamped"""
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
        node = UltraPrecisePerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Perception node stopped by user")
    except Exception as e:
        print(f"Perception error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Perception node shutdown complete.")


if __name__ == '__main__':
    main()
