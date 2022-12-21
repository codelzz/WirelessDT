# WirelessDT: A Digital Twin Platform for Real-Time Evaluation of Wireless Software Applications

WirelessDT is a **Wireless** **D**igital **T**win Platform for Wireless Software Applications (WSAs) Evaluation.

In this [ICSE2023 DEMO Video](https://youtu.be/9Kl-3jgMBUA), we evaluate a wireless indoor localisation mobile application with two typical prediction algorithms: 1) Kalman Filter-based Trilateration and 2) Deep Recurrent Neural Network, as a case study to demonstrate the capabilities of WirelessDT. 

## Abstract

Wireless technology has become one of the most important parts of our daily routine. Besides being used for communication, the wireless signal has been applied to various Wireless Software Applications (WSAs). The signal fluctuation caused by the measurement system or the environmental dynamic can lead to nonnegligible influences on WSA performance. However, quantitative and qualitative real-time evaluation of WSAs is challenging due to limited resources and intractable practical issues. To overcome these challenges, we propose WirelsssDT, a wireless digital twin platform, using digital twin and real-time ray tracing technologies to emulate the signal propagation and generate emulation data for real-time WSAs evaluation.

## Getting Start

*NOTICES:* WirelessDT can only be executed on Window Systems with NVIDIA RTX GPU. The suggested GPU hardware is RTX3070 or RTX4090.

To use WirelessDT, we need to clone both WirelessDT and the WiTracing Unreal Engine (UE) repositories.

### STEP 1: Clone WiTracing Unreal Engine Project.

Unreal Engine is a private project and is only visible to its subscribers. First, We need to associate the GitHub account with our UE account. For the step for account association, please refer to [How do I link my Unreal Engine account with my Github account?](https://www.epicgames.com/help/en-US/epic-accounts-c5719348850459/connect-accounts-c5719351300507/how-do-i-link-my-unreal-engine-account-with-my-github-account-a5720369784347)

After associating the accounts, we should have access to our [Unreal Engine Fork](https://github.com/codelzz/UnrealEngine/tree/dev/ue5-witracing) with our built-in WiTracing Engine. Then you can clone the engine code from `dev/ue5-witracing` branch.

```shell
git clone https://github.com/codelzz/UnrealEngine.git
```

After cloning the UE source code, you can refer to the official page [Downloading Unreal Engine Source Code](https://docs.unrealengine.com/5.1/en-US/downloading-unreal-engine-source-code/) for engine setup. After setup the UE source code, we need to compile it to generate the binary files.

### STEP2: Clone WirelessDT Project

```shell
git clone https://github.com/codelzz/WirelessDT.git
```

After the project is cloned, we need to compile the project. 

1. Right click the `.uproject`, then select  `Switch Unreal Engine Version...`, and choose the one with `Source build ....`. This will associate `WirelessDT` with `WiTracing Unreal Engine`. 
2. Right click the `.uproject`  again and select `Generate Visual Studio project file`. The solution file end with `.sln` should appear.
3. Open the `.sln` file and compile the project.

### STEP3: Run the Digital Twin Platform

After compiling, we can double-click the  `.uproject`  file to launch the platform. The `DemoOffice_30mx30m` map is the default game level which can be executed without Physical Twin. (You can change to the other map from the content fold window under All>Content>Demo>Map. The `DemoCorridor_Calibration` level is the calibration environment that required Physical Twin.)

Now, you can click `Play` button in the Editor to start the platform.

## Advanced Settings

### WiTracing Engine

We provide configuration of WiTracing Engine through Unreal Engine Editor Command Line Interface. Press <**\`**> button to activate the Command Line Interface, Then type in the prefix `r.WiTracing.` for available commands:

| Command                    | Description                                                  | Type  |
| -------------------------- | ------------------------------------------------------------ | ----- |
| r.WiTracing.TXPower        | The transmit power of the transmitter (default 25 dBm)       | Int   |
| r.WiTracing.Wavelength     | The wavelength of the transmit signal (default 0.125 m for 2.4 GHz) | Float |
| r.WiTracing.RXSensitivity  | The receiver sensitivity (default -95 dBm)                   | Int   |
| r.WiTracing.SamplePerPixel | The number of WiTracing samples for each pixel per frame     | Int   |
| r.WiTracing.MaxBounce      | The maximum number of bounces for each WiTracing ray         | Int   |

### Data Sharing API Endpoint

To change the endpoint for data sharing, we can configure the setting of IP and Port in `WiTracingAgent` from the `UE Editor Outliner` window. The data is transmitted in JSON format.

### Auto-Spawn AI

The number of auto-spawn AI can be changed by the `Number of AIs` in the AISpawn actor. This will impact the number of randomly walking characters in runtime.

### Target AI Path

The path of the target AI carrying the wireless receiver can be adjusted by changing the trajectory of the `AISpline` Actor.

Besides the above settings, we open-source our project and provide total freedom for other researchers or developers to customize their virtual environment with UE Editor.

## Extra Setup for DEMO Reproduction.

The physical twin hardware and the localization application are required to reproduce the demo.

Here is detailed settings and environment:

#### Physical Twin Hardware Components

| Component       | Hardware                               |
| --------------- | -------------------------------------- |
| Wireless Sensor | Seeed Studio XIAO nRF52840             |
| Location Sensor | Intel® RealSense™ Tracking Camera T265 |
| Mainboard       | Raspberry Pi 4 Model B                 |
| Power Supply    | 5V/3A Power bank                       |

The python script for physical twin state synchronization is located at `\Source\WiTracingHCI` . To run the hardware, we need to upload the script in Raspberry Pi 4 Model B, then execute it by 

```shell
sudo python app.py
```

(The Arduino program for XIAO nRF52840 is in `\Source\WiTracingHCI\WiTracingBLE\RX`)

#### Platform Deployment Environment

| Component | Setup                          |
| --------- | ------------------------------ |
| System    | Windows 10 Pro                 |
| Processor | Intel(R) i7-10870H CPU         |
| RAM       | 16.0 GB                        |
| GPU       | NVIDIA GeForce RTX 3070 Laptop |

#### WiTracing Parameter

| Parameter                          | Value                  |
| ---------------------------------- | ---------------------- |
| Wireless Signal Wavelength         | 0.125m (e.g. 2.4GHz)   |
| Wireless Transmit Power            | 25 dBm                 |
| Wireless System Gain               | -45 dB                 |
| Wireless Receiver Sensitivity      | -95 dBm                |
| WiTracing Resolution               | 16,384 Pixel (128x128) |
| Maximum WiTracing Sample Per Pixel | 1                      |
| Maximum WiTracing Bounces          | 3                      |

#### Positioning Application

To clone the repository.

```shell
git clone https://github.com/codelzz/WirelessDTApp.git
```

The application has been tested on iPhone 12 Pro Max with iOS 16.1



## Troubles Shooting

* If you encounter any issues while executing, please check your GPU driver whether is appropriately installed. To verify the driver, you can create an empty unreal engine project with ray-tracing enabled and run it to see whether the issue still exists.
