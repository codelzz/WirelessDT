#include <ArduinoBLE.h>

BLEService service("00000000-0000-0000-0000-000000000000");
BLEIntCharacteristic characteristic("00000000-0000-0000-0000-000000000000", BLERead | BLEBroadcast);

String GLocalName = "WiBLE";
int Interval = 0x50;
int HalfInterval = Interval / 2;

void setup() {
  // SET PIN
  pinMode(LEDG, OUTPUT);

  // Serial.begin(9600);
  if (!BLE.begin()) {
    while (1);
  }

  service.addCharacteristic(characteristic);
  BLE.addService(service);
  BLE.setAdvertisingInterval(Interval); // in the unit of 0.625ms, the minimal we can set is 20ms (0x20) | 100ms (0xA0), 1000 (0x3e8)
  BLEAdvertisingData scanData;
  scanData.setLocalName(GLocalName.c_str());
  BLE.setScanResponseData(scanData);
  BLE.setAdvertisedServiceUuid("00000000-0000-0000-0000-000000000000");
  BLE.advertise();
  digitalWrite(LEDG, LOW);
}

void flash()
{
  digitalWrite(LEDG, HIGH);
  delay(HalfInterval);
  digitalWrite(LEDG, LOW);
  delay(HalfInterval);
}

void loop() {
  // delay Here won't affect the speed of advertising, they are in different thread
  // if (Serial)
  // {
    // digitalWrite(LEDG, HIGH);
    // Serial.println(BLE.address());
    // delay(0xff);
  // }
  flash();
}