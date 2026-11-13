// Compile the approved reasoning-layer headers into pchie_core (hooks idle, flags off).
#include "pchie/rl/RlCommon.h"
#include "pchie/rl/RlFlags.h"
#include "pchie/rl/HashMoveRecord.h"
#include "pchie/rl/RlSideChain.h"
#include "pchie/rl/UntrustedTokens.h"
#include "pchie/rl/LeafValueSource.h"   // L1
#include "pchie/rl/PlanMemory.h"        // L2
#include "pchie/rl/ForkScore.h"         // L3
#include "pchie/rl/BeliefState.h"       // L4
#include "pchie/rl/CrutchAudit.h"       // L4
#include "pchie/rl/Options.h"           // L5
#include "pchie/rl/SymbolicPlanner.h"   // L6
#include "pchie/rl/LabelSchema.h"       // L6
#include "pchie/rl/PchStackRlHooks.h"
#include "pchie/PchStack.h"

namespace pchie { namespace rl {
// Flags-off identity used by the regression: default flags are official-clean and a default PchStack is idle.
bool rl98_flags_off_identity() {
  PchStack s;
  return official_runner_clean(official_defaults()) && s.rl_idle();
}
} }
