#include <ArduinoBLE.h>

BLEService service("00000000-0000-0000-0000-000000000000");
BLEIntCharacteristic characteristic("00000000-0000-0000-0000-000000000000", BLERead | BLEBroadcast);

const uint8_t completeRawAdvertisingData[] = {0x02,0x01,0x06,0x09,0xff,0x01,0x01,0x00,0x01,0x02,0x03,0x04,0x05};

// Global Index: when TX start, it will sent n-rount non-positive for RX to initialize statistic cache
int GIndex = -1; 
String GLocalName = "WiBLE";

void setup() {
  // SET PIN
  pinMode(LEDG, OUTPUT);
  pinMode(LEDR, OUTPUT);

  Serial.begin(9600);
  // while (!Serial);
  if (!BLE.begin()) {
    while (1);
  }
 
  service.addCharacteristic(characteristic);
  BLE.addService(service);
  
  // Build advertising data packet
  // BLEAdvertisingData advData;
  // If a packet has a raw data parameter, then all the other parameters of the packet will be ignored
  // advData.setRawData(completeRawAdvertisingData, sizeof(completeRawAdvertisingData));  
  // Copy set parameters in the actual advertising packet
  // BLE.setAdvertisingData(advData);
  BLE.setAdvertisingInterval(0x20); // in the unit of 0.625ms, the minimal we can set 20ms
}

void loop() {
  BLEAdvertisingData scanData;
  String msg = BLE.address() + "|" + String(GIndex++);
  Serial.println("[Send] >>> " + msg);
  scanData.setLocalName(msg.c_str());
  // Copy set parameters in the actual scan response packet
  BLE.setScanResponseData(scanData);
  BLE.advertise();
  if (GIndex <= 0)
  {
    digitalWrite(LEDR, LOW);
    delay(0x4ff); // 0xff = 255 * 4
    digitalWrite(LEDR, HIGH);
  } 
  else
  {
    digitalWrite(LEDG, LOW);
    delay(0x1);  // for flashing
    digitalWrite(LEDG, HIGH);
  }
  BLE.stopAdvertise();
  delay(0x20);
}

