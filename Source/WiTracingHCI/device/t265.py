import pyrealsense2 as rs
from udpthread.runnable import Runnable
import utils
import math


class T265Proxy(Runnable):
    def __init__(self, on_data_recv_fn, wait_time=0.0):
        super(T265Proxy, self).__init__(wait_time=wait_time)
        self.on_data_recv_fn = on_data_recv_fn
        self.pipe = rs.pipeline()
        self.config = rs.config()
        self.config.enable_all_streams()
        self.payload = {}

    def start(self):
        # Ref: Configure the option
        # https://github.com/IntelRealSense/librealsense/issues/1011
        # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.option.html#pyrealsense2.option
        self.profile = self.config.resolve(self.pipe)
        self.sensor = self.profile.get_device().first_pose_sensor()
        self.sensor.set_option(rs.option.enable_mapping, True)
        self.sensor.set_option(rs.option.enable_relocalization, True)
        self.sensor.set_option(rs.option.enable_pose_jumping, True)
        self.sensor.set_option(rs.option.enable_dynamic_calibration, True)
        self.sensor.set_option(rs.option.enable_map_preservation, True)
        self.profile = self.pipe.start(self.config)
        super().start()

    def stop(self):
        super().stop()
        self.pipe.stop()

    def do(self):
        try:
            frames = self.pipe.wait_for_frames()
            frame = frames.get_pose_frame()
            if frame:
                # parse data
                pose = frame.get_pose_data()
                self.update_payload(pose)
                self.on_data_recv_fn(self.payload)
        except:
            print('[ERR] T265 failed')
            self.pipe.stop()
            self.pipe.start()
            print('[INF] Restart T265 pipe')
        finally:
            pass

    def update_payload(self, pose):
        # convert to UE5 coordinate system
        # ref: https://github.com/IntelRealSense/librealsense/blob/master/doc/t265.md
        pitch, roll, yaw = self.parse_rotation(pose)
        self.payload = {
            'timestamp':utils.millisecond(),
            'x':-pose.translation.z,
            'y':pose.translation.x,
            'z':pose.translation.y,
            'vx':-pose.velocity.z,
            'vy':pose.velocity.x,
            'vz':pose.velocity.y,
            'ax':-pose.acceleration.z,
            'ay':pose.acceleration.x,
            'az':pose.acceleration.y,
            'pitch': pitch,
            'roll': roll,
            'yaw': yaw,
        }

    @staticmethod
    def parse_rotation(pose):
        w = pose.rotation.w
        x = -pose.rotation.z
        y = pose.rotation.x
        z = -pose.rotation.y
        pitch =  -math.asin(2.0 * (x*z - w*y)) * 180.0 / math.pi;
        roll  =  math.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z) * 180.0 / math.pi;
        yaw   =  math.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z) * 180.0 / math.pi;
        return pitch, roll, yaw