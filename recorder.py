import os
import json
import time
import threading
import io
import numpy as np
from PIL import Image
import cv2
import zmq

class Recorder:
    def __init__(self, client, output_dir="recordings"):
        self.client = client
        self.output_dir = output_dir
        self.is_recording = False
        self.thread = None
        self.frame_count = 0
        self.session_dir = None

        # 🛡️ Create an independent, dedicated ZMQ SUB socket for the recorder thread
        # This prevents the recorder and controller from competing for messages on client.camera
        self.camera_sub = client.ctx.socket(zmq.SUB)
        self.camera_sub.setsockopt(zmq.RCVHWM, 2)
        self.camera_sub.connect(f"tcp://{client.host}:{client.port_base + 3}")
        self.camera_sub.setsockopt(zmq.SUBSCRIBE, b"")

    def start(self):
        """Start recording the camera stream"""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, f"session_{timestamp_str}")
        self.depth_dir = os.path.join(self.session_dir, "depth")
        os.makedirs(self.depth_dir, exist_ok=True)
        
        self.metadata_path = os.path.join(self.session_dir, "metadata.jsonl")
        self.rgb_video_path = os.path.join(self.session_dir, "rgb_video.mp4")
        self.depth_video_path = os.path.join(self.session_dir, "depth_video.mp4")
        self.cam_info_path = os.path.join(self.session_dir, "camera_info.json")

        # Save camera calibration & extrinsics once for offline back-projection
        try:
            cam_info, _ = self.client.call("get_camera_info")
            with open(self.cam_info_path, "w", encoding="utf-8") as f:
                json.dump(cam_info, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save camera_info: {e}")
        
        self.is_recording = True
        self.thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.thread.start()

    def _recording_loop(self):
        rgb_writer = None
        depth_writer = None
        metadata_file = open(self.metadata_path, "w", encoding="utf-8")
        
        try:
            while self.is_recording:
                try:
                    # Fetch latest camera multipart message from dedicated recorder socket ("latest wins")
                    msg = self.camera_sub.recv_multipart()
                    while self.camera_sub.poll(0):
                        msg = self.camera_sub.recv_multipart()
                    
                    if not msg or len(msg) < 3:
                        time.sleep(0.01)
                        continue

                    header_bytes, jpg_bytes, raw_bytes = msg[0], msg[1], msg[2]
                    header = json.loads(header_bytes.decode("utf-8") if isinstance(header_bytes, bytes) else header_bytes)
                    
                    seq = header["seq"]
                    t = header["t"]
                    pose = header["pose"]
                    width = header["width"]
                    height = header["height"]
                    
                    # Process and write RGB frame to MP4 video
                    img_rgb = np.array(Image.open(io.BytesIO(jpg_bytes)))
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Process Depth frame into Depth Video
                    depth = np.frombuffer(raw_bytes, dtype="<u2").reshape(height, width)

                    # Normalize depth (0 to 6000 mm) to 0-255 range for 8-bit video visualization
                    depth_norm = np.clip(depth / 6000.0 * 255.0, 0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                    if rgb_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        rgb_writer = cv2.VideoWriter(self.rgb_video_path, fourcc, 15.0, (width, height))
                        depth_writer = cv2.VideoWriter(self.depth_video_path, fourcc, 15.0, (width, height))

                    rgb_writer.write(img_bgr)
                    depth_writer.write(depth_color)
                    
                    # Write synchronization metadata record for offline processing
                    meta_entry = {
                        "seq": seq,
                        "t": t,
                        "pose": pose,
                        "width": width,
                        "height": height
                    }
                    metadata_file.write(json.dumps(meta_entry) + "\n")
                    metadata_file.flush()
                    
                    self.frame_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Error in camera recording loop: {e}")
                    time.sleep(0.02)
                    
        finally:
            if rgb_writer is not None:
                rgb_writer.release()
            if depth_writer is not None:
                depth_writer.release()
            metadata_file.close()

    def stop(self):
        """Stop the camera recording thread and finalize files."""
        if not self.is_recording:
            return
        self.is_recording = False
        if self.thread:
            self.thread.join(timeout=2.0)
        try:
            self.camera_sub.close()
        except Exception:
            pass
        print(f"🛑 Camera recording stopped. Total frames recorded: {self.frame_count}")
        print(f"files saved in: {self.session_dir}")