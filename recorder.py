""" Isolated background recorder thread capturing RGB/Depth Stream and synchonization metadata"""

import os
import json
import time
import threading
import io
import numpy as np
from PIL import Image
import cv2

class Recorder:
    def __init__(self, client, output_dir="recordings"):
        self.client = client
        self.output_dir = output_dir
        self.is_recording = False
        self.thread = None
        self.frame_count = 0
        self.session_dir = None

        client.create_recorder_camera_socket()

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
                    header, jpg, raw = self.client.latest_recording()

                    header = json.loads(header)
                    
                    seq = header["seq"]
                    t = header["t"]
                    pose = header["pose"]
                    width = header["width"]
                    height = header["height"]
                    
                    # Process and write RGB frame to MP4 video
                    img_rgb = np.array(Image.open(io.BytesIO(jpg)))
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Process Depth frame into Depth Video
                    depth = np.frombuffer(raw, dtype="<u2").reshape(height, width)

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
            if self.client.camera_recorder:
                self.client.camera_recorder.close()
        except Exception:
            pass
        print(f"🛑 Camera recording stopped. Total frames recorded: {self.frame_count}")
        print(f"files saved in: {self.session_dir}")