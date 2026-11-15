#include "pchie/research/AsciiObservation.h"

#include "pchie/research/Digest.h"

#include <cstdio>

namespace pchie {
namespace research {

namespace {

bool str_field(const MiniJson& p, const char* key, std::string& out, std::string& err) {
  const MiniJson* v = p.get(key);
  if (!v || v->type != MiniJson::Type::String) { err = std::string("ascii view: missing text field '") + key + "'"; return false; }
  out = v->str;
  return true;
}

bool int_field(const MiniJson& p, const char* key, int lo, int hi, int& out, std::string& err) {
  const MiniJson* v = p.get(key);
  if (!v || v->type != MiniJson::Type::Number || v->number < lo || v->number > hi || double(int(v->number)) != v->number) {
    err = std::string("ascii view: bad number field '") + key + "'";
    return false;
  }
  out = int(v->number);
  return true;
}

bool is_lower_hex(const std::string& s, size_t n) {
  if (s.size() != n) return false;
  for (char c : s)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}

// rows lines of cols characters each, joined by '\n', none of them a newline or a control character
bool shaped(const std::string& s, int cols, int rows) {
  if (s.size() != size_t(cols + 1) * size_t(rows) - 1) return false;
  for (int y = 0; y < rows; ++y) {
    const size_t base = size_t(y) * size_t(cols + 1);
    for (int x = 0; x < cols; ++x) {
      const unsigned char c = static_cast<unsigned char>(s[base + size_t(x)]);
      if (c < 0x20 || c > 0x7e) return false;
    }
    if (y + 1 < rows && s[base + size_t(cols)] != '\n') return false;
  }
  return true;
}

}  // namespace

bool capture_from_json(const MiniJson& p, AsciiCapture& out, std::string& err) {
  out = AsciiCapture{};
  if (p.type != MiniJson::Type::Object) { err = "ascii view: result is not an object"; return false; }
  if (!str_field(p, "format", out.format, err) || !str_field(p, "view", out.view, err) || !str_field(p, "ramp", out.ramp, err) ||
      !str_field(p, "palette", out.palette, err) || !str_field(p, "lut_sha", out.lut_sha, err) || !str_field(p, "glyph", out.glyph, err) ||
      !str_field(p, "fg", out.fg, err) || !str_field(p, "depth", out.depth, err) || !str_field(p, "cls", out.cls, err) ||
      !str_field(p, "sha256", out.sha256, err))
    return false;
  if (!int_field(p, "cols", 1, 256, out.cols, err) || !int_field(p, "rows", 1, 256, out.rows, err)) return false;
  const MiniJson* r = p.get("rle");
  if (!r || r->type != MiniJson::Type::Bool) { err = "ascii view: missing field 'rle'"; return false; }
  out.rle = r->boolean;
  if (const MiniJson* cam = p.get("camera")) {
    out.cam_yaw = cam->num_or("yaw", 0.0);
    out.cam_pitch = cam->num_or("pitch", 0.0);
  }
  return true;
}

std::string rle_encode(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size();) {
    const char c = text[i];
    size_t j = i + 1;
    if (c != '\n')
      while (j < text.size() && text[j] == c) ++j;
    const size_t run = j - i;
    if (run >= 4) {
      out += c;
      out += std::to_string(run);
    } else {
      out.append(run, c);
    }
    i = j;
  }
  return out;
}

bool rle_decode(const std::string& rle, std::string& out) {
  out.clear();
  for (size_t i = 0; i < rle.size();) {
    const char c = rle[i];
    if (c >= '0' && c <= '9') return false;  // a count with no character before it
    size_t j = i + 1;
    size_t n = 0;
    while (j < rle.size() && rle[j] >= '0' && rle[j] <= '9') {
      n = n * 10 + size_t(rle[j] - '0');
      if (n > (1u << 20)) return false;
      ++j;
    }
    if (j == i + 1) {
      out += c;
    } else {
      if (n < 4 || c == '\n') return false;  // not a form the engine makes
      out.append(n, c);
    }
    i = j;
  }
  return true;
}

std::string canonical_record(int cols, int rows, const std::string& view, const std::string& lut16, const std::string& glyph,
                             const std::string& fg, const std::string& depth, const std::string& cls) {
  return "nemo_ascii.v1|" + std::to_string(cols) + "|" + std::to_string(rows) + "|" + view + "|ramp.v1|pal16.v1|" + lut16 + "|" +
         glyph + "|" + fg + "|" + depth + "|" + cls;
}

bool verify_capture(const AsciiCapture& c, AsciiView& out, std::string& err) {
  out = AsciiView{};
  if (c.format != "nemo_ascii.v1") { err = "ascii view: unknown format"; return false; }
  if (c.ramp != "ramp.v1" || c.palette != "pal16.v1") { err = "ascii view: unknown ramp or palette"; return false; }
  if (c.view != "objects" && c.view != "pixels_depth") { err = "ascii view: unknown view mode"; return false; }
  if (!is_lower_hex(c.lut_sha, 16)) { err = "ascii view: bad depth table id"; return false; }
  if (!is_lower_hex(c.sha256, 64)) { err = "ascii view: bad record hash"; return false; }
  std::string glyph;
  if (c.rle) {
    if (!rle_decode(c.glyph, glyph)) { err = "ascii view: bad run-length code"; return false; }
  } else {
    glyph = c.glyph;
  }
  for (char ch : glyph)
    if (ch >= '0' && ch <= '9') { err = "ascii view: digits in the glyph rows"; return false; }
  if (!shaped(glyph, c.cols, c.rows) || !shaped(c.fg, c.cols, c.rows) || !shaped(c.depth, c.cols, c.rows) ||
      !shaped(c.cls, c.cols, c.rows)) {
    err = "ascii view: channel shape does not match cols x rows";
    return false;
  }
  const std::string want = sha256_hex(canonical_record(c.cols, c.rows, c.view, c.lut_sha, glyph, c.fg, c.depth, c.cls));
  if (want != c.sha256) { err = "ascii view: record hash mismatch (fail closed)"; return false; }
  out.view = c.view;
  out.glyph = std::move(glyph);
  out.fg = c.fg;
  out.depth = c.depth;
  out.cls = c.cls;
  out.sha256 = c.sha256;
  out.cols = c.cols;
  out.rows = c.rows;
  out.cam_yaw = c.cam_yaw;
  out.cam_pitch = c.cam_pitch;
  return true;
}

bool fit_glyph(const AsciiView& v, int cols, int rows, std::string& out, std::string& err) {
  out.clear();
  if (cols <= 0 || rows <= 0 || cols > v.cols || rows > v.rows) { err = "ascii view: smaller than its slot"; return false; }
  const int x0 = (v.cols - cols) / 2, y0 = (v.rows - rows) / 2;
  for (int y = 0; y < rows; ++y) {
    if (y) out += '\n';
    out.append(v.glyph, size_t(y0 + y) * size_t(v.cols + 1) + size_t(x0), size_t(cols));
  }
  return true;
}

std::vector<AsciiRequest> ascii_requests(uint32_t ctx) {
  std::vector<AsciiRequest> r;
  const ContextPlan p = context_plan(ctx);
  if (!p.fits) return r;
  for (const ViewSlot& s : p.views) {
    AsciiRequest q;
    q.slot = s.name;
    q.rel = (s.name.find("left") != std::string::npos) ? 3 : (s.name.find("right") != std::string::npos) ? 1 : 0;
    q.fovea = s.name.rfind("fovea", 0) == 0;
    q.cols = 40;
    q.fit_cols = s.cols;
    q.fit_rows = s.rows;
    r.push_back(q);
  }
  return r;
}

const char* ascii_legend() {
  return "legend: # wall, | door, k key, * light, o pit, E exit, & creature, ~ liquid; other characters are brightness";
}

std::string observation_text(const std::string& label, const std::string& glyph_rows, int cols, int rows) {
  return "view " + label + " (" + std::to_string(cols) + "x" + std::to_string(rows) + "):\n" + glyph_rows + "\n";
}

bool log_view(SideLog& log, const std::string& tag, const std::string& slot, const AsciiView& v, std::string& err) {
  if (!log.enabled()) return true;
  char cam[96];
  std::snprintf(cam, sizeof(cam), "\"yaw_deg\":%.1f,\"pitch_deg\":%.1f", v.cam_yaw, v.cam_pitch);
  const std::string body = "{\"format\":\"pchie.ascii_obs.v1\",\"tag\":" + MiniJson::quote(tag) + ",\"slot\":" + MiniJson::quote(slot) +
                           ",\"cols\":" + std::to_string(v.cols) + ",\"rows\":" + std::to_string(v.rows) + ",\"view\":" +
                           MiniJson::quote(v.view) + ",\"record_sha256\":\"" + v.sha256 + "\",\"glyph_sha256\":\"" + sha256_hex(v.glyph) +
                           "\"," + cam;
  return log.append(body, err);
}

}  // namespace research
}  // namespace pchie
