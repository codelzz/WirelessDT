from abc import ABCMeta, abstractmethod
import time
import threading


class IRunnable(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        """"""

    @abstractmethod
    def start(self):
        """"""

    @abstractmethod
    def stop(self):
        """"""


class Runnable(IRunnable):
    bStopping = 0
    thread = None
    wait_time = 0.01

    def __init__(self, wait_time=0.01):
        self.wait_time = wait_time
        self.thread = threading.Thread(target=self.run, daemon=True)

    # ~ IRunnable interface
    def start(self):
        self.bStopping = False
        self.thread.start()

    def stop(self):
        self.bStopping = True

    def run(self):
        while not self.bStopping:
            self.do()
            time.sleep(self.wait_time)

    def do(self):
        pass
    # ~ IRunnable interface end

    def print(self, payload):
        print(f"[{self.__class__.__name__}] {payload}")

    def __repr__(self):
        return f"<{self.__class__.__name__}>"