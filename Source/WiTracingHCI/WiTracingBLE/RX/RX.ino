/**
 * RX 
 * 
 * This program scan for Bluetooth® Low Energy peripherals and collect statistic data.
 *
 * REFERENCE:
 * ScanCallback.ino
 */

#include <ArduinoBLE.h>

long int PacketCounter = 0;
long int StartTimeMS = 0;
// String Uuid = "35828EE5-8390-43AE-8457-C4C6BBC1B255";
// String Uuid = "00000000-0000-0000-0000-000000000000";
String Uuid = "fec4"; // BlueCat Beacon

void setup() {
  pinMode(LEDG, OUTPUT);
  // enable serial
  Serial.begin(9600);
  while (!Serial);
  // begin initialization
  if (!BLE.begin()) {
    Serial.println("[ERROR] starting BLE module failed!");
    while (1);
  }
  // set the discovered event handle
  StartTimeMS = millis();
  BLE.setEventHandler(BLEDiscovered, onBLEDiscovered);
  // BLE.scan(true);
  BLE.scanForUuid(Uuid,true);
  //BLE.scanForAddress(Addr, true);
}

void loop() {
  BLE.poll(100);
}

void onBLEDiscovered(BLEDevice peripheral)
{
  // -----------------------------------------------------------------------------
  // |  TXAddress | RSSI | 
  // -----------------------------------------------------------------------------
  digitalWrite(LEDG, LOW);
  if (Serial)
  {
    // peripheral.advertisedServiceUuid() + "," + 
    String data = peripheral.address() + "," + peripheral.rssi();
    Serial.println(data);
  }
  digitalWrite(LEDG, HIGH);
}
