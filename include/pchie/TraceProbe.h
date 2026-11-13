// Bolt PCHIE - one labeled unit-test / fixture probe for the headless grader.
// Faces follow the Multiplexor ruling (south=in1, west=in2, east=control, north=out).
// Dividor uses south as input and expected_north as the expected output.
// Never a golden; never the official answer key.
#pragma once

namespace pchie {

struct TraceProbe {
  const char* component = "";
  int south = 0;
  int west = 0;
  int east = 0;
  int expected_north = 0;
};

}  // namespace pchie
