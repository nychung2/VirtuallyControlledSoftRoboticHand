#include "programmable_air.h"
void setup() {
  // put your setup code here, to run once:
  initializePins();
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'l') {
      blow();
      switchOnPump(2,100);
      delay(2000);
      switchOffPump(2);
      closeAllValves();
      delay(2000);
      vent();
    }
  }
}
