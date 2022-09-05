## DeepWiSim

### Goal

This project aims to utilize UE5 path-tracing shader to simulate wireless signal propagation.

In the next version, we need to make shader allow disable pdf calculation which consumes too much resource and slow down the whole system

### Main Component

#### Simulator (UE)

* WiTracingAgent


### Improvement
1. Improve RSSI Plot code quality (add comment)
1. Allow switch RX between digital twin and player pawn


### Known Bugs
1. Digital Twin program randomly hang after run for a while.

```bash
Exception in thread Thread-4 (run):
Traceback (most recent call last):
  File "/usr/lib/python3.10/threading.py", line 1009, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.10/threading.py", line 946, in run
    self._target(*self._args, **self._kwargs)
  File "/home/x/App/WiTracing/thread/runnable.py", line 40, in run
    self.do()
  File "/home/x/App/WiTracing/device/t265.py", line 36, in do
    frames = self.pipe.wait_for_frames()
RuntimeError: Frame didn't arrive within 5000
```