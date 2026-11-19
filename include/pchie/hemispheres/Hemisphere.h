// Bolt PCHIE - the hemisphere interface, and the seam for weight edits.
//
// A hemisphere is anything that can give a distribution over the legal moves of a
// world state. The toy policy in Mjolnir, a dense transformer, a mixture-of-experts
// model or a Mamba hybrid all fit: nothing here assumes one model's layer layout.
//
// Weight edits go through IWeightOverlayTarget, keyed by the canonical (Hugging Face)
// tensor names, so a converter or an editor never needs to know how a runtime lays a
// model out. A runtime maps those names to its own slots (Gemma4OverlayTarget does it
// for the E4B forward pass); a MoE or Mamba runtime adds its own target, with expert or
// state-space tensors under their own canonical names.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#pragma once
#include "pchie/hemispheres/SampleSet.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
class Gemma4Model;
struct QTensor;

namespace hemispheres {

struct HemiState {
    const int8_t* features = nullptr;   // what a small policy sees
    uint32_t n_features = 0;
    uint32_t n_actions = 0;
    uint32_t legal = 0;                 // bit a set: move a is legal
    const void* rich = nullptr;         // what a large model reads (text, a captured trunk state, frames)
};

class IHemisphere {
public:
    virtual ~IHemisphere() = default;
    // probs[0..n_actions) = p * 65535 for the legal moves, 0 for the rest (see to_q16).
    virtual bool distribution(const HemiState& s, uint16_t probs[kMaxActions], std::string& err) = 0;
    virtual std::array<uint8_t, kFingerprint> fingerprint() const = 0;
    virtual std::string name() const = 0;
};

// A model whose linear weights can be pointed at other memory, by canonical name.
class IWeightOverlayTarget {
public:
    virtual ~IWeightOverlayTarget() = default;
    // rows = outputs, cols = inputs (row-major, inputs contiguous), as in safetensors.
    virtual bool shape(const std::string& name, uint64_t& rows, uint64_t& cols) const = 0;
    // True if a change to this tensor takes effect (a GPU copy made at load time would not).
    virtual bool editable(const std::string& name) const = 0;
    // Point the tensor at BF16 data (rows x cols). The caller keeps the data alive.
    virtual bool set_bf16(const std::string& name, const uint16_t* data, std::string& err) = 0;
    // Back to the weights the model was loaded with.
    virtual void restore_all() = 0;
};

// Gemma 4 (E4B) names: model.language_model.layers.N.{self_attn.{q,k,v,o}_proj,
// mlp.{gate,up,down}_proj, per_layer_input_gate, per_layer_projection}.weight.
// Only layers >= first_editable_layer are editable (the CUDA backend copies the
// layers below it to the GPU at init).
class Gemma4OverlayTarget final : public IWeightOverlayTarget {
public:
    Gemma4OverlayTarget(Gemma4Model& m, int first_editable_layer);
    bool shape(const std::string& name, uint64_t& rows, uint64_t& cols) const override;
    bool editable(const std::string& name) const override;
    bool set_bf16(const std::string& name, const uint16_t* data, std::string& err) override;
    void restore_all() override;
    // "model.language_model.layers.38.mlp.up_proj.weight" -> layer 38, slot Up. False if not a linear weight.
    static bool parse_name(const std::string& name, int& layer, int& slot);
private:
    Gemma4Model& m_;
    int first_;
    struct Saved { int layer; int slot; const void* data; uint32_t type; uint64_t ne0, ne1; };
    std::vector<Saved> saved_;
};

// float <-> bfloat16, round to nearest even (NaN kept quiet).
uint16_t to_bf16(float f);
float from_bf16(uint16_t h);

} // namespace hemispheres
} // namespace pchie
