import pyrealsense2 as rs
from thread.runnable import Runnable
import utils


class T265Proxy(Runnable):
    def __init__(self, on_data_recv_fn, wait_time=0.0):
        super(T265Proxy, self).__init__(wait_time=wait_time)
        self.on_data_recv_fn = on_data_recv_fn
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.pose)

    def start(self):
        self.pipe.start(self.config)
        super().start()

    def stop(self):
        super().stop()
        self.pipe.stop()

    def do(self):
        try:
            # Wait for the next set of frames from the camera
            frames = self.pipe.wait_for_frames()
            pose = frames.get_pose_frame()

            if pose:
                # pack data with current timestamp in millisecond
                data = (utils.millisecond(), pose)
                self.on_data_recv_fn(data)
        finally:
            pass