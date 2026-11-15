// Nemo View observations for Little Nemo: the vision.capture_ascii result checked and turned into prompt text.
// The capture is untrusted input. It is accepted only when its shape is right, its glyph rows carry no digits
// and its SHA-256 matches the engine's canonical record recomputed here, so a view cannot be forged or
// silently cut. Every accepted view is written to its own side chain (never the official chain).
#pragma once

#include "pchie/MiniJson.h"
#include "pchie/research/ContextBudget.h"
#include "pchie/research/SideLog.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace research {

/// The fields of one vision.capture_ascii result, as received (glyph may be run-length coded).
struct AsciiCapture {
  std::string format, view, ramp, palette, lut_sha, glyph, fg, depth, cls, sha256;
  int cols = 0;
  int rows = 0;
  bool rle = false;
  double cam_yaw = 0.0;
  double cam_pitch = 0.0;
};

/// A checked view: glyph rows decoded, every channel the same shape, record hash verified.
struct AsciiView {
  std::string view, glyph, fg, depth, cls, sha256;
  int cols = 0;
  int rows = 0;
  double cam_yaw = 0.0;
  double cam_pitch = 0.0;
};

/// Reads the MCP result object. Missing or mistyped fields fail closed.
bool capture_from_json(const MiniJson& p, AsciiCapture& out, std::string& err);

/// The engine's run-length code: a run of 4 or more becomes <char><count>; newlines are never coded.
std::string rle_encode(const std::string& text);
bool rle_decode(const std::string& rle, std::string& out);

/// nemo_ascii.v1|cols|rows|view|ramp.v1|pal16.v1|<lut16>|glyph|fg|depth|cls
std::string canonical_record(int cols, int rows, const std::string& view, const std::string& lut16, const std::string& glyph,
                             const std::string& fg, const std::string& depth, const std::string& cls);

/// Decodes and checks a capture. Any mismatch fails closed with a reason.
bool verify_capture(const AsciiCapture& c, AsciiView& out, std::string& err);

/// The centre cols x rows of the glyph rows (fails when the view is smaller than the slot).
bool fit_glyph(const AsciiView& v, int cols, int rows, std::string& out, std::string& err);

/// One request per view slot of a context plan: the tool's cols and region for that slot.
struct AsciiRequest {
  std::string slot;
  int rel = 0;  // 0 forward, 1 right, 3 left (relative to his heading)
  int cols = 40;
  bool fovea = false;
  int fit_cols = 0;
  int fit_rows = 0;
};
/// ctx 1024: three 32x12 foveas. ctx 2048: a 40x22 forward view and 32x12 foveas left and right.
std::vector<AsciiRequest> ascii_requests(uint32_t ctx);

/// The legend line every ASCII observation starts with (untrusted observation text, no coordinates).
const char* ascii_legend();

/// The observation text for one view slot: a header line and the glyph rows.
std::string observation_text(const std::string& label, const std::string& glyph_rows, int cols, int rows);

/// Side chain line for one accepted view.
bool log_view(SideLog& log, const std::string& tag, const std::string& slot, const AsciiView& v, std::string& err);

}  // namespace research
}  // namespace pchie
