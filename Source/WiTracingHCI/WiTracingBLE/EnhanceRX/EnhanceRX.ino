/**
 * # RX 
 * 
 * This program scan for Bluetooth® Low Energy peripherals and collect statistic data.
 *
 * ## Feature
 * 1. PacketTransmissionSpeed - estimation of packet TX speed
 * 2. PacketReceptionSpeed   - estimation of packet RX Speed
 * 
 */

#include <ArduinoBLE.h>

#define _DEBUG_ false

// const arduino::String TXAddress = "23:c9:cc:4b:a0:3a";
const String TXAddress = "75:ac:a7:b3:2e:45";
const int WARNUP_PACKET_INDEX_THRESHOLD = 0;  // the TX will send n-round non-positive index for RX initalize statistic cache

long int GStartTime = 0;      // ms
long int GLastTime = 0;
int GPacketStartIndex = -1;
int GReceivedPacketCount = 0;

void setup() {
  // PIN
  pinMode(LEDG, OUTPUT);
  
  // enable serial
  Serial.begin(9600);
  // begin initialization
  if (!BLE.begin()) {
    while (1);
  }
  // set the discovered event handle
  ResetStatisticMetric();
  BLE.setEventHandler(BLEDiscovered, onBLEDiscovered);
  BLE.scan(true);
}

void loop() {
  // poll the central for events
  BLE.poll();
}

void onBLEDiscovered(BLEDevice peripheral)
{
  if (!peripheral.hasLocalName())
  {
    return;
  }

  digitalWrite(LEDG, HIGH);
  if (IsMatchedDevice(peripheral))
  {
    digitalWrite(LEDG, LOW);
    GReceivedPacketCount++;
    int PacketIndex = GetPacketIndex(peripheral);
    if (NeedResetStatisticMetric(PacketIndex))
    {
      ResetStatisticMetric();
      // When the TX is reboot, the packet Index will restart from zero
      // When detecting index less than zero do reset
      return; // Warning up for reset
    }
    //
    long int Timestamp = millis();
    long int TotalTimeElapsed = GetTotalTimeElapsed(Timestamp);
    long int LastTimeElapsed = GetLastTimeElapsed(Timestamp);
    float TXSpeed = EstimatePacketTransmissionSpeed(PacketIndex, TotalTimeElapsed);
    float RXSpeed = EstimatePacketReceptionSpeed(TotalTimeElapsed);

    String msg = "";
    if (_DEBUG_)
    {
      msg = "ID:" + String(PacketIndex) + "," 
      + "RSSI:" + String(peripheral.rssi()) + "," 
      + "RX Count:" + String(GReceivedPacketCount) + "," 
      + "RX Spd:" + String(RXSpeed) + "," 
      + "TX Spd:" + String(TXSpeed) + "," 
      + "Last Eps:" + String(LastTimeElapsed) + "," 
      + "Total Eps:" + String(TotalTimeElapsed) + "," 
      + "Timestamp:" + String(Timestamp);
    }
    else 
    {
            
    // Massage Formate:
    // ---------------------------------------------------------------------------------------------------------------------------------------
    // |  TXAddress | PacketIndex | RSSI | Received Packet Count | RX Speed | TX Speed | LastTimeElapsed | TotalTimeElasped | Timestamp (ms) |
    // ---------------------------------------------------------------------------------------------------------------------------------------
      msg = String(peripheral.address()) + ","
      + String(PacketIndex) + "," 
      + String(peripheral.rssi()) + "," 
      + String(GReceivedPacketCount) + "," 
      + String(RXSpeed) + "," 
      + String(TXSpeed) + "," 
      + String(LastTimeElapsed) + "," 
      + String(TotalTimeElapsed) + "," 
      + String(Timestamp);
    }
    
    // msg.replace('|',',');
    Serial.println(msg);
    digitalWrite(LEDG, HIGH);
  }
}

bool NeedResetStatisticMetric(int PacketIndex)
{
  return PacketIndex <= WARNUP_PACKET_INDEX_THRESHOLD;
}

bool IsMatchedDevice(BLEDevice peripheral)
{
  return peripheral.localName().indexOf(peripheral.address())>=0;
}

void ResetStatisticMetric()
{
  GStartTime = 0;
  GPacketStartIndex = -1;
  GReceivedPacketCount = 0;
  GStartTime = millis();
}

long int GetLastTimeElapsed(long int Timestamp)
{
  long int TimeElapsed = Timestamp - GLastTime;
  GLastTime = Timestamp;
  return TimeElapsed;
}

long int GetTotalTimeElapsed(long int Timestamp)
{
  return Timestamp - GStartTime;
}

int GetPacketIndex(BLEDevice peripheral)
{
  // Get Packet Index from received packet
   return peripheral.localName().substring(18).toInt();
}

float EstimatePacketTransmissionSpeed(int PacketIndex, long int TimeElapsed)
{
  // Init global packet start index cache
  if (GPacketStartIndex < 0)
  {
    GPacketStartIndex = PacketIndex;
  }
  // estimate speed
  return float(TimeElapsed) / (PacketIndex - GPacketStartIndex + 1);
}

float EstimatePacketReceptionSpeed(long int TimeElapsed)
{
  return float(TimeElapsed) / GReceivedPacketCount;
}






