import json
import zmq

class Client:
    def __init__(self, host="localhost", port_base=5555):
        self.host = host
        self.port_base = port_base

        self.ctx = zmq.Context()

        self.rpc = self.ctx.socket(zmq.REQ)
        self.rpc.connect(self.url(0))

        self.cmd = self.ctx.socket(zmq.PUSH)
        self.cmd.connect(self.url(1))
        
        self.state = self.ctx.socket(zmq.SUB)
        self.state.connect(self.url(2))
        self.state.setsockopt(zmq.SUBSCRIBE, b"")

        self.camera = self.ctx.socket(zmq.SUB)
        self.camera.setsockopt(zmq.RCVHWM, 2)
        self.camera.connect(self.url(3))
        self.camera.setsockopt(zmq.SUBSCRIBE, b"")

    def url(self, k):
        return f"tcp://{self.host}:{self.port_base + k}"

    def call(self, method, **params):
        self.rpc.send_json({"method": method, "params": params})
        frames = self.rpc.recv_multipart()
        rep = json.loads(frames[0])
        if not rep["ok"]:
            raise RuntimeError(rep["error"])
        return rep["result"], frames[1:]

    def latest_state(self):
        """Most recent message for robot state: drop anything queued (streams are 'latest wins')."""
        msg = self.state.recv_multipart()
        while self.state.poll(0):
            msg = self.state.recv_multipart()
        return json.loads(msg[0])

    def latest_camera(self):
        """Most recent message for camera: drop anything queued (streams are 'latest wins')."""
        msg = self.camera.recv_multipart()
        while self.camera.poll(0):
            msg = self.camera.recv_multipart()
        return msg

    def send_goto(self, goal_id, x, y, yaw=None):
        """Send a goto command."""
        payload = {"type": "goto", "id": goal_id, "x": x, "y": y}
        if yaw is not None:
            payload["yaw"] = yaw
        self.cmd.send_json(payload)

    def send_wheel(self, left, right):
        """Send a wheel command."""
        payload = {"type": "wheel", "left": left, "right": right}
        self.cmd.send_json(payload)