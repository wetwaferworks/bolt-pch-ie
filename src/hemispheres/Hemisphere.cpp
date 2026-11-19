// Bolt PCHIE - weight overlays by canonical name. See include/pchie/hemispheres/Hemisphere.h.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#include "pchie/hemispheres/Hemisphere.h"
#include "pchie/pchie_gemma4.h"
#include <cstring>

namespace pchie {
namespace hemispheres {

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu)) return uint16_t((u >> 16) | 0x0040u);
    const uint32_t round = 0x7FFFu + ((u >> 16) & 1u);
    return uint16_t((u + round) >> 16);
}

float from_bf16(uint16_t h) {
    const uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool Gemma4OverlayTarget::parse_name(const std::string& name, int& layer, int& slot) {
    static const char* prefix = "model.language_model.layers.";
    static const struct { const char* tail; Gemma4Model::LinearSlot slot; } tails[] = {
        {".self_attn.q_proj.weight", Gemma4Model::LinearSlot::Q},
        {".self_attn.k_proj.weight", Gemma4Model::LinearSlot::K},
        {".self_attn.v_proj.weight", Gemma4Model::LinearSlot::V},
        {".self_attn.o_proj.weight", Gemma4Model::LinearSlot::O},
        {".mlp.gate_proj.weight", Gemma4Model::LinearSlot::Gate},
        {".mlp.up_proj.weight", Gemma4Model::LinearSlot::Up},
        {".mlp.down_proj.weight", Gemma4Model::LinearSlot::Down},
        {".per_layer_input_gate.weight", Gemma4Model::LinearSlot::InpGate},
        {".per_layer_projection.weight", Gemma4Model::LinearSlot::Proj},
    };
    const size_t pl = std::strlen(prefix);
    if (name.size() <= pl || name.compare(0, pl, prefix) != 0) return false;
    size_t i = pl;
    int n = 0, digits = 0;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9' && digits < 4) { n = n * 10 + (name[i] - '0'); ++i; ++digits; }
    if (digits == 0 || (digits > 1 && name[pl] == '0')) return false;
    const std::string tail = name.substr(i);
    for (const auto& t : tails)
        if (tail == t.tail) { layer = n; slot = int(t.slot); return true; }
    return false;
}

Gemma4OverlayTarget::Gemma4OverlayTarget(Gemma4Model& m, int first_editable_layer) : m_(m), first_(first_editable_layer) {}

bool Gemma4OverlayTarget::shape(const std::string& name, uint64_t& rows, uint64_t& cols) const {
    int l = 0, s = 0;
    QTensor q;
    if (!parse_name(name, l, s) || !m_.linear(l, Gemma4Model::LinearSlot(s), q)) return false;
    rows = q.ne1;
    cols = q.ne0;
    return true;
}

bool Gemma4OverlayTarget::editable(const std::string& name) const {
    int l = 0, s = 0;
    QTensor q;
    return parse_name(name, l, s) && l >= first_ && m_.linear(l, Gemma4Model::LinearSlot(s), q);
}

bool Gemma4OverlayTarget::set_bf16(const std::string& name, const uint16_t* data, std::string& err) {
    int l = 0, s = 0;
    QTensor cur;
    if (!parse_name(name, l, s) || !m_.linear(l, Gemma4Model::LinearSlot(s), cur)) { err = "unknown tensor " + name; return false; }
    if (l < first_) { err = name + " runs on the GPU in this setup; a change would not take effect"; return false; }
    bool have = false;
    for (const Saved& v : saved_) have = have || (v.layer == l && v.slot == s);
    if (!have) saved_.push_back(Saved{l, s, cur.data, cur.type, cur.ne0, cur.ne1});
    QTensor t;
    t.data = data;
    t.type = 30;
    t.ne0 = cur.ne0;
    t.ne1 = cur.ne1;
    return m_.set_linear(l, Gemma4Model::LinearSlot(s), t, err);
}

void Gemma4OverlayTarget::restore_all() {
    for (const Saved& v : saved_) {
        QTensor t;
        t.data = v.data;
        t.type = v.type;
        t.ne0 = v.ne0;
        t.ne1 = v.ne1;
        std::string err;
        m_.set_linear(v.layer, Gemma4Model::LinearSlot(v.slot), t, err);
    }
    saved_.clear();
}

} // namespace hemispheres
} // namespace pchie
