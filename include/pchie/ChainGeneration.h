// Bolt PCHIE - chain generation switch. The default is the existing SHA-256 chain (v1); the ratcheted
// chain (v2) is OFF unless a caller asks for it by name. Existing call sites that build Sha256HashChain
// directly are untouched, so no existing chain or run hash can move because of this file.
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include "pchie/IHashChain.h"
#include "pchie/RatchetHashChain.h"
#include "pchie/Sha256HashChain.h"

namespace pchie {

enum class ChainGeneration : uint32_t { Sha256V1 = 1, RatchetV2 = 2 };

inline constexpr ChainGeneration kDefaultChainGeneration = ChainGeneration::Sha256V1;

inline const char* chain_generation_id(ChainGeneration g) {
  return g == ChainGeneration::RatchetV2 ? kRatchetGenerationId : "pchie.seal.v1-sha256";
}

inline ChainGeneration parse_chain_generation(const std::string& s) {
  if (s.empty() || s == "v1" || s == "pchie.seal.v1-sha256") return ChainGeneration::Sha256V1;
  if (s == "v2" || s == kRatchetGenerationId || s == kRatchetSaltedGenerationId) return ChainGeneration::RatchetV2;
  throw std::invalid_argument("unknown chain generation: " + s);
}

// The ratcheted generation needs its genesis (weights, run id, harness version, and optionally a secret salt); v1 ignores it.
inline std::unique_ptr<IHashChain> make_hash_chain(ChainGeneration g = kDefaultChainGeneration, const RatchetGenesis* genesis = nullptr) {
  if (g == ChainGeneration::RatchetV2) {
    if (!genesis) throw std::invalid_argument("ratcheted chain needs a genesis");
    return std::make_unique<RatchetHashChain>(*genesis);
  }
  return std::make_unique<Sha256HashChain>();
}

}  // namespace pchie
