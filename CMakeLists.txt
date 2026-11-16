cmake_minimum_required(VERSION 3.16)
project(bolt_pchie CXX)
option(PCHIE_CUDA "Build the CUDA backend" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

add_library(pchie_core STATIC
  src/pchie_tree.cpp src/pchie_kv_cow.cpp src/pchie_q4_det.cpp src/pchie_rollout.cpp src/pchie_host.cpp src/pchie_gguf.cpp
  src/pchie_kv_pools.cpp src/pchie_gemma4.cpp src/pchie_mcts.cpp
  src/LogitStats.cpp src/LogProbLeafEvaluator.cpp src/TurnBoundaryTerminalPolicy.cpp
  src/FirstPlayUrgency.cpp src/RootNoise.cpp src/CommitSelector.cpp
  src/Sha256.cpp src/OsRandom.cpp
  src/GgufWeightsHash.cpp
  src/NemotronModel.cpp
  src/rl/RlFlags.cpp src/rl/LabelSchema.cpp
  src/rl/pchie_rl98_layers.cpp)
target_include_directories(pchie_core PUBLIC include)
target_link_libraries(pchie_core PUBLIC Threads::Threads)
if(WIN32)
  target_link_libraries(pchie_core PUBLIC bcrypt)  # OS random generator for TESLA seeds and signing keys
endif()

# AVX2 kernels. AUTO (the default) turns them on only when the compiler accepts the flag and the build
# machine's CPU runs AVX2 code; ON and OFF force the choice. Without AVX2 the kernels take their portable
# scalar path; check the maze hashes on such a build before comparing it with an AVX2 one.
set(PCHIE_AVX2 AUTO CACHE STRING "Build the AVX2 kernels: AUTO, ON or OFF")
set_property(CACHE PCHIE_AVX2 PROPERTY STRINGS AUTO ON OFF)
if(MSVC)
  set(_pchie_avx2_flag /arch:AVX2)
else()
  set(_pchie_avx2_flag -mavx2)
endif()
set(_pchie_use_avx2 OFF)
if(PCHIE_AVX2 STREQUAL "ON")
  set(_pchie_use_avx2 ON)
elseif(PCHIE_AVX2 STREQUAL "AUTO")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(${_pchie_avx2_flag} PCHIE_COMPILER_HAS_AVX2)
  if(PCHIE_COMPILER_HAS_AVX2 AND NOT CMAKE_CROSSCOMPILING)
    set(_pchie_probe ${CMAKE_CURRENT_BINARY_DIR}/pchie_avx2_probe.cpp)
    file(WRITE ${_pchie_probe} [=[
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
static bool os_and_cpu_avx2() {
  int r[4]; __cpuid(r, 0); if (r[0] < 7) return false;
  __cpuid(r, 1); const bool osxsave = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
  if (!osxsave || !avx || (_xgetbv(0) & 6) != 6) return false;
  __cpuidex(r, 7, 0); return (r[1] & (1 << 5)) != 0;
}
#else
static bool os_and_cpu_avx2() { __builtin_cpu_init(); return __builtin_cpu_supports("avx2"); }
#endif
int main() {
  if (!os_and_cpu_avx2()) return 1;
  volatile int a = 3; __m256i v = _mm256_set1_epi32(a); v = _mm256_add_epi32(v, v);
  return _mm256_extract_epi32(v, 0) == 6 ? 0 : 1;
}
]=])
    try_run(_pchie_avx2_run _pchie_avx2_compile ${CMAKE_CURRENT_BINARY_DIR}/pchie_avx2_probe ${_pchie_probe}
            COMPILE_DEFINITIONS ${_pchie_avx2_flag})
    if(_pchie_avx2_compile AND "${_pchie_avx2_run}" STREQUAL "0")
      set(_pchie_use_avx2 ON)
    endif()
  endif()
elseif(NOT PCHIE_AVX2 STREQUAL "OFF")
  message(FATAL_ERROR "PCHIE_AVX2 must be AUTO, ON or OFF")
endif()
message(STATUS "PCHIE AVX2 kernels: ${_pchie_use_avx2} (PCHIE_AVX2=${PCHIE_AVX2})")
if(_pchie_use_avx2)
  set(_pchie_arch_opt ${_pchie_avx2_flag})
else()
  set(_pchie_arch_opt "")
endif()
# Determinism: no FMA contraction, no fast-math.
if(MSVC)
  target_compile_options(pchie_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/fp:precise ${_pchie_arch_opt} /W4>)
else()
  target_compile_options(pchie_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off -fno-fast-math ${_pchie_arch_opt} -Wall -Wextra>)
endif()

enable_testing()
add_executable(pchie_spine_test tests/pchie_spine_test.cpp)
target_link_libraries(pchie_spine_test PRIVATE pchie_core)
add_test(NAME pchie_spine COMMAND pchie_spine_test)
add_executable(pchie_timing_handshake_test tests/pchie_timing_handshake_test.cpp)
target_link_libraries(pchie_timing_handshake_test PRIVATE pchie_core)
add_test(NAME pchie_timing_handshake COMMAND pchie_timing_handshake_test)
add_executable(pchie_search_rules_test tests/pchie_search_rules_test.cpp)
target_link_libraries(pchie_search_rules_test PRIVATE pchie_core)
add_test(NAME pchie_search_rules COMMAND pchie_search_rules_test)
add_executable(pchie_strict_standin_gate_test tests/pchie_strict_standin_gate_test.cpp)
target_link_libraries(pchie_strict_standin_gate_test PRIVATE pchie_core)
add_test(NAME pchie_strict_standin_gate COMMAND pchie_strict_standin_gate_test)
add_executable(pchie_hash_chain_test tests/pchie_hash_chain_test.cpp)
target_link_libraries(pchie_hash_chain_test PRIVATE pchie_core)
add_test(NAME pchie_hash_chain COMMAND pchie_hash_chain_test)
add_executable(pchie_nemotron_test tests/pchie_nemotron_test.cpp)
target_link_libraries(pchie_nemotron_test PRIVATE pchie_core)
add_test(NAME pchie_nemotron COMMAND pchie_nemotron_test)
add_executable(pchie_pch_gate_test tests/pchie_pch_gate_test.cpp)
target_link_libraries(pchie_pch_gate_test PRIVATE pchie_core)
add_test(NAME pchie_pch_gate COMMAND pchie_pch_gate_test)
add_executable(pchie_cpu_leaf_path_test tests/pchie_cpu_leaf_path_test.cpp)
target_link_libraries(pchie_cpu_leaf_path_test PRIVATE pchie_core)
add_test(NAME pchie_cpu_leaf_path COMMAND pchie_cpu_leaf_path_test)
add_executable(pchie_htn_domain_test tests/pchie_htn_domain_test.cpp)
target_link_libraries(pchie_htn_domain_test PRIVATE pchie_core)
add_test(NAME pchie_htn_domain COMMAND pchie_htn_domain_test)
add_executable(pchie_htn_planner_test tests/pchie_htn_planner_test.cpp)
target_link_libraries(pchie_htn_planner_test PRIVATE pchie_core)
add_test(NAME pchie_htn_planner COMMAND pchie_htn_planner_test)
add_executable(pchie_mcts_search_test tests/pchie_mcts_search_test.cpp)
target_link_libraries(pchie_mcts_search_test PRIVATE pchie_core)
add_test(NAME pchie_mcts_search COMMAND pchie_mcts_search_test)
add_executable(pchie_dualpool_stepback_test tests/pchie_dualpool_stepback_test.cpp)
target_link_libraries(pchie_dualpool_stepback_test PRIVATE pchie_core)
add_test(NAME pchie_dualpool_stepback COMMAND pchie_dualpool_stepback_test)
add_executable(pchie_ncm_candidate_test tests/pchie_ncm_candidate_test.cpp)
target_link_libraries(pchie_ncm_candidate_test PRIVATE pchie_core)
add_test(NAME pchie_ncm_candidate COMMAND pchie_ncm_candidate_test)
add_executable(pchie_pch_stack_test tests/pchie_pch_stack_test.cpp)
target_link_libraries(pchie_pch_stack_test PRIVATE pchie_core)
add_test(NAME pchie_pch_stack COMMAND pchie_pch_stack_test)
add_executable(pchie_htn_precondition_test tests/pchie_htn_precondition_test.cpp)
target_link_libraries(pchie_htn_precondition_test PRIVATE pchie_core)
add_test(NAME pchie_htn_precondition COMMAND pchie_htn_precondition_test)
add_executable(pchie_ncm_join_test tests/pchie_ncm_join_test.cpp)
target_link_libraries(pchie_ncm_join_test PRIVATE pchie_core)
add_test(NAME pchie_ncm_join COMMAND pchie_ncm_join_test)
add_executable(pchie_joined_stepback_test tests/pchie_joined_stepback_test.cpp)
target_link_libraries(pchie_joined_stepback_test PRIVATE pchie_core)
add_test(NAME pchie_joined_stepback COMMAND pchie_joined_stepback_test)
# BOS policy and opt-in early stop after identical failed attempts.
add_executable(pchie_bos_policy_test tests/pchie_bos_policy_test.cpp)
target_link_libraries(pchie_bos_policy_test PRIVATE pchie_core)
add_test(NAME pchie_bos_policy COMMAND pchie_bos_policy_test)
add_executable(pchie_identical_retry_stop_test tests/pchie_identical_retry_stop_test.cpp)
target_link_libraries(pchie_identical_retry_stop_test PRIVATE pchie_core)
add_test(NAME pchie_identical_retry_stop COMMAND pchie_identical_retry_stop_test)
# Observation contract: raw path (parse_special=false) and BOS hardening.
add_executable(pchie_obs_contract_test tests/pchie_obs_contract_test.cpp)
target_link_libraries(pchie_obs_contract_test PRIVATE pchie_core)
add_test(NAME pchie_obs_contract COMMAND pchie_obs_contract_test)
# Failed-actuation retry: opt-in Stage B injection with tabu and symbolic fallback. Official runners leave this off.
add_executable(pchie_far_retry_test tests/pchie_far_retry_test.cpp)
target_link_libraries(pchie_far_retry_test PRIVATE pchie_core)
add_test(NAME pchie_far_retry COMMAND pchie_far_retry_test)
add_executable(pchie_maze_test tests/pchie_maze_test.cpp)
target_link_libraries(pchie_maze_test PRIVATE pchie_core)
add_test(NAME pchie_maze COMMAND pchie_maze_test)
add_executable(pchie_determinism_gauntlet_test tests/pchie_determinism_gauntlet_test.cpp)
target_link_libraries(pchie_determinism_gauntlet_test PRIVATE pchie_core)
add_test(NAME pchie_determinism_gauntlet COMMAND pchie_determinism_gauntlet_test)
add_executable(pchie_freeze_recovery_test tests/pchie_freeze_recovery_test.cpp)
target_link_libraries(pchie_freeze_recovery_test PRIVATE pchie_core)
add_test(NAME pchie_freeze_recovery COMMAND pchie_freeze_recovery_test)
add_executable(pchie_prompt_injection_test tests/pchie_prompt_injection_test.cpp)
target_link_libraries(pchie_prompt_injection_test PRIVATE pchie_core)
add_test(NAME pchie_prompt_injection COMMAND pchie_prompt_injection_test)
add_executable(pchie_budget_timeout_test tests/pchie_budget_timeout_test.cpp)
target_link_libraries(pchie_budget_timeout_test PRIVATE pchie_core)
add_test(NAME pchie_budget_timeout COMMAND pchie_budget_timeout_test)
add_executable(pchie_wander_regression_test tests/pchie_wander_regression_test.cpp)
target_link_libraries(pchie_wander_regression_test PRIVATE pchie_core)
add_test(NAME pchie_wander_regression COMMAND pchie_wander_regression_test)
add_executable(pchie_pit_fixture_test tests/pchie_pit_fixture_test.cpp)
target_link_libraries(pchie_pit_fixture_test PRIVATE pchie_core)
add_test(NAME pchie_pit_fixture COMMAND pchie_pit_fixture_test)
add_executable(pchie_mirror_sim_gate_test tests/pchie_mirror_sim_gate_test.cpp)
target_link_libraries(pchie_mirror_sim_gate_test PRIVATE pchie_core)
add_test(NAME pchie_mirror_sim_gate COMMAND pchie_mirror_sim_gate_test)
add_executable(pchie_scaled_pit_maze_test tests/pchie_scaled_pit_maze_test.cpp)
target_link_libraries(pchie_scaled_pit_maze_test PRIVATE pchie_core)
add_test(NAME pchie_scaled_pit_maze COMMAND pchie_scaled_pit_maze_test)
add_executable(pchie_trace_grader_test tests/pchie_trace_grader_test.cpp)
target_link_libraries(pchie_trace_grader_test PRIVATE pchie_core)
add_test(NAME pchie_trace_grader COMMAND pchie_trace_grader_test)
add_executable(pchie_trace_curriculum_test tests/pchie_trace_curriculum_test.cpp)
target_link_libraries(pchie_trace_curriculum_test PRIVATE pchie_core)
add_test(NAME pchie_trace_curriculum COMMAND pchie_trace_curriculum_test)
add_executable(pchie_stageb_door_test tests/pchie_stageb_door_test.cpp)
target_link_libraries(pchie_stageb_door_test PRIVATE pchie_core)
add_test(NAME pchie_stageb_door COMMAND pchie_stageb_door_test)
# LithoKraft pilot: the reasoning model used directly inside the game (view-only examples).
add_executable(pchie_lithokraft_pilot_test tests/pchie_lithokraft_pilot_test.cpp)
target_link_libraries(pchie_lithokraft_pilot_test PRIVATE pchie_core)
add_test(NAME pchie_lithokraft_pilot COMMAND pchie_lithokraft_pilot_test)
if(WIN32)
  add_executable(pchie_lk_pilot tools/pchie_lk_pilot.cpp)
  target_link_libraries(pchie_lk_pilot PRIVATE pchie_core)
  # Vision-only navigation (Scratch), CPU decide path. Memory receipts are on by default (--neml off).
  add_executable(pchie_lk_visnav vision/pchie_lk_visnav.cpp)
  target_include_directories(pchie_lk_visnav PRIVATE vision)
  target_link_libraries(pchie_lk_visnav PRIVATE pchie_core pchie_research user32)  # research: --obs ascii only
endif()
add_executable(pchie_gguf_probe tests/pchie_gguf_probe.cpp)
target_link_libraries(pchie_gguf_probe PRIVATE pchie_core)
add_executable(pchie_gguf_dump tests/pchie_gguf_dump.cpp)
target_link_libraries(pchie_gguf_dump PRIVATE pchie_core)
add_executable(pchie_kernels_test tests/pchie_kernels_test.cpp)
target_link_libraries(pchie_kernels_test PRIVATE pchie_core)
add_test(NAME pchie_kernels COMMAND pchie_kernels_test)
add_executable(pchie_generate tests/pchie_generate.cpp)
target_link_libraries(pchie_generate PRIVATE pchie_core)

# CUDA backend. Determinism: no FMA contraction, IEEE div/sqrt, no fast math.
if(PCHIE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(pchie_cuda STATIC src/pchie_cuda.cu)
    target_include_directories(pchie_cuda PUBLIC include)
    set_target_properties(pchie_cuda PROPERTIES CUDA_STANDARD 17)
    target_compile_options(pchie_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false -prec-div=true -prec-sqrt=true -ftz=false -Xcompiler=/fp:precise>)
    target_link_libraries(pchie_cuda PUBLIC pchie_core CUDA::cudart)
    add_executable(pchie_cuda_gate tests/pchie_cuda_gate.cpp)
    target_link_libraries(pchie_cuda_gate PRIVATE pchie_cuda)
    add_executable(pchie_fork_batching tests/pchie_fork_batching.cpp)
    target_link_libraries(pchie_fork_batching PRIVATE pchie_cuda)
    add_executable(pchie_search_gate tests/pchie_search_gate.cpp)
    target_link_libraries(pchie_search_gate PRIVATE pchie_cuda)
    # Backend guards on tiny hand-built models (needs a GPU; reports "skipped" without one).
    add_executable(pchie_cuda_backend_guard_test tests/pchie_cuda_backend_guard_test.cpp)
    target_link_libraries(pchie_cuda_backend_guard_test PRIVATE pchie_cuda)
    add_test(NAME pchie_cuda_backend_guard COMMAND pchie_cuda_backend_guard_test)
    set_tests_properties(pchie_cuda_backend_guard PROPERTIES LABELS cuda)
  else()
    message(STATUS "PCHIE_CUDA: no CUDA compiler found, skipping the CUDA backend")
  endif()
endif()

# CUDA backend shape limits, checked on the host (no GPU or CUDA compiler needed).
add_executable(pchie_cuda_limits_test tests/pchie_cuda_limits_test.cpp)
target_link_libraries(pchie_cuda_limits_test PRIVATE pchie_core)
add_test(NAME pchie_cuda_limits COMMAND pchie_cuda_limits_test)

# Deterministic exp and rounding on NaN, inf and out-of-range input (and bit-identical results otherwise).
add_executable(pchie_det_math_test tests/pchie_det_math_test.cpp)
target_link_libraries(pchie_det_math_test PRIVATE pchie_core)
add_test(NAME pchie_det_math COMMAND pchie_det_math_test)
add_executable(pchie_crypto_util_test tests/pchie_crypto_util_test.cpp)
target_link_libraries(pchie_crypto_util_test PRIVATE pchie_core)
add_test(NAME pchie_crypto_util COMMAND pchie_crypto_util_test)

add_executable(pchie_bounds_test tests/pchie_bounds_test.cpp)
target_link_libraries(pchie_bounds_test PRIVATE pchie_core)
add_test(NAME pchie_bounds COMMAND pchie_bounds_test)

# Join-order invariant: deterministic fold by stable work id, plus a shuffle-arrival test.
add_executable(pchie_join_order_test tests/pchie_join_order_test.cpp)
target_link_libraries(pchie_join_order_test PRIVATE pchie_core)
add_test(NAME pchie_join_order COMMAND pchie_join_order_test)

# Gate split: the gate sees proposal tokens only.
add_executable(pchie_gate_split_test tests/pchie_gate_split_test.cpp)
target_link_libraries(pchie_gate_split_test PRIVATE pchie_core)
add_test(NAME pchie_gate_split COMMAND pchie_gate_split_test)

# Reasoning-layer contract and flags tests, flags off.
add_executable(pchie_rl_contract_test tests/rl/pchie_rl_contract_test.cpp)
target_link_libraries(pchie_rl_contract_test PRIVATE pchie_core)
target_compile_definitions(pchie_rl_contract_test PRIVATE PCHIE_REPO_ROOT="${CMAKE_SOURCE_DIR}")
add_test(NAME pchie_rl_contract COMMAND pchie_rl_contract_test)
add_executable(pchie_rl_flags_test tests/rl/pchie_rl_flags_test.cpp)
target_link_libraries(pchie_rl_flags_test PRIVATE pchie_core)
add_test(NAME pchie_rl_flags COMMAND pchie_rl_flags_test)

# Research layers (opt-in, --research-* flags; all off by default). A separate library: pchie_core is unchanged,
# and the maze runner (pchie_rl_l1l2_test --maze) only uses it when a research flag is given. No model, about a second.
add_library(pchie_research STATIC
  src/research/ResearchFlags.cpp src/research/ExperienceMap.cpp src/research/GoExplore.cpp
  src/research/ConstraintSolver.cpp src/research/StepScorer.cpp src/research/TreeOfThoughts.cpp
  src/research/Reflexion.cpp src/research/FrozenDynamics.cpp src/research/ContextBudget.cpp
  src/research/SideLog.cpp src/research/MazeAdvisor.cpp src/research/AsciiObservation.cpp src/research/OwnMapPlans.cpp)
target_link_libraries(pchie_research PUBLIC pchie_core)
add_executable(pchie_research_test tests/research/pchie_research_test.cpp)
target_link_libraries(pchie_research_test PRIVATE pchie_research)
target_compile_definitions(pchie_research_test PRIVATE PCHIE_RESEARCH_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/research/data")
add_test(NAME pchie_research COMMAND pchie_research_test --unit)
# Freezes a dynamics model from research side-chain logs (no model, seconds).
add_executable(pchie_research_tool tools/pchie_research_tool.cpp)
target_link_libraries(pchie_research_tool PRIVATE pchie_research)

# Harness (scratch research, off the official path): Inspect-style tasks, solvers, scorers and transcripts,
# and one plug interface for many games (BALROG-style).
# Its own library; pchie_core and every existing hash are unchanged. No model, about a second.
add_library(pchie_harness STATIC
  src/harness/Transcript.cpp src/harness/Task.cpp src/harness/Eval.cpp src/harness/GamePlug.cpp src/harness/Plugs.cpp
  src/harness/SkillChecks.cpp)
target_link_libraries(pchie_harness PUBLIC pchie_research)
add_executable(pchie_harness_test tests/harness/pchie_harness_test.cpp tests/harness/task_eval_test.cpp tests/harness/plugs_test.cpp
  tests/harness/skills_test.cpp)
target_link_libraries(pchie_harness_test PRIVATE pchie_harness)
add_test(NAME pchie_harness COMMAND pchie_harness_test --unit --dir ${CMAKE_CURRENT_BINARY_DIR})
# Plug server for the thin Gymnasium adapter in tools/gym (off-path: no chain, no keys, no network).
add_executable(pchie_plug_server tools/pchie_plug_server.cpp)
target_link_libraries(pchie_plug_server PRIVATE pchie_harness)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME pchie_gym_adapter COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/gym/test_pchie_gym.py
           --server $<TARGET_FILE:pchie_plug_server>)
endif()

# L1/L2/sealed-tuple flags-on ablations (decide path only, CPU).
add_executable(pchie_rl_l1l2_test tests/rl/pchie_rl_l1l2_test.cpp)
target_link_libraries(pchie_rl_l1l2_test PRIVATE pchie_core pchie_research)  # research: --research-* on maze runs only
add_test(NAME pchie_rl_l1l2 COMMAND pchie_rl_l1l2_test --unit)

# Readers for finished planner-comparison and query-before-plan run logs
# (stop-and-report rule, L2 engagement) and the L2 store path guard. No model, seconds.
add_executable(pchie_rl_run_compare_test tests/rl/pchie_rl_run_compare_test.cpp)
target_link_libraries(pchie_rl_run_compare_test PRIVATE pchie_core)
add_test(NAME pchie_rl_run_compare COMMAND pchie_rl_run_compare_test)

# Loop guard v2, the shared reasoning-layer guard, alone and inside every maze driver (on by default; off = old results). No model.
add_executable(pchie_loop_guard_test tests/rl/pchie_loop_guard_test.cpp)
target_link_libraries(pchie_loop_guard_test PRIVATE pchie_core)
add_test(NAME pchie_loop_guard COMMAND pchie_loop_guard_test)
# The Stage B harness's unit section only (no model, well under a second). The full run with real Little Nemo stays out of CTest.
add_test(NAME pchie_stageb_unit COMMAND pchie_stageb_door_test --unit)

# Ratcheted seal chain (chain generation v2, off by default; header only, nothing added to pchie_core).
add_executable(pchie_ratchet_chain_test tests/pchie_ratchet_chain_test.cpp)
target_link_libraries(pchie_ratchet_chain_test PRIVATE pchie_core)
add_test(NAME pchie_ratchet_chain COMMAND pchie_ratchet_chain_test)
# Epoch re-root at the harness's real fork and resume points (ratcheted chains only; v1 chains unchanged).
add_executable(pchie_chain_epoch_test tests/pchie_chain_epoch_test.cpp)
target_link_libraries(pchie_chain_epoch_test PRIVATE pchie_core)
add_test(NAME pchie_chain_epoch COMMAND pchie_chain_epoch_test)

# TESLA-manner stream authentication (RFC 4082, chain order as the clock) and attestation
# appraisal (RFC 9334). ON by default in the maze drivers since 2026-11-13; --tesla off and
# --attest off turn each one off for testing. The watcher is a separate program. This CTest
# never talks to the TPM: without a quote the verdict is Unattested, never Attested.
add_executable(pchie_tesla_watcher tools/pchie_tesla_watcher.cpp)
target_link_libraries(pchie_tesla_watcher PRIVATE pchie_core)
add_executable(pchie_gguf_validate_test tests/pchie_gguf_validate_test.cpp)
target_link_libraries(pchie_gguf_validate_test PRIVATE pchie_core)
add_test(NAME pchie_gguf_validate COMMAND pchie_gguf_validate_test)
add_executable(pchie_gemma4_load_validate_test tests/pchie_gemma4_load_validate_test.cpp)
target_link_libraries(pchie_gemma4_load_validate_test PRIVATE pchie_core)
add_test(NAME pchie_gemma4_load_validate COMMAND pchie_gemma4_load_validate_test)
add_executable(pchie_tesla_stream_test tests/pchie_tesla_stream_test.cpp)
target_link_libraries(pchie_tesla_stream_test PRIVATE pchie_core)
add_test(NAME pchie_tesla_stream COMMAND pchie_tesla_stream_test --watcher $<TARGET_FILE:pchie_tesla_watcher>)

# Transparency anchoring side tool and its checks. The tool reads finished-run artifacts; no run links it.
# The verifier test makes throwaway keys and logs under the build folder and removes them on a clean pass.
add_executable(pchie_anchor anchoring/src/pchie_anchor.cpp anchoring/src/pchie_anchor_fs.cpp)
target_link_libraries(pchie_anchor PRIVATE pchie_core)
add_executable(pchie_anchor_unit_test anchoring/tests/anchor_unit.cpp)
target_include_directories(pchie_anchor_unit_test PRIVATE anchoring/src)
target_link_libraries(pchie_anchor_unit_test PRIVATE pchie_core)
add_test(NAME pchie_anchor_unit COMMAND pchie_anchor_unit_test)
add_executable(pchie_anchor_verifier_test anchoring/tests/anchor_verifier_test.cpp anchoring/src/pchie_anchor_fs.cpp)
target_include_directories(pchie_anchor_verifier_test PRIVATE anchoring/src)
target_link_libraries(pchie_anchor_verifier_test PRIVATE pchie_core)
add_test(NAME pchie_anchor_verifier COMMAND pchie_anchor_verifier_test ${CMAKE_CURRENT_BINARY_DIR}/anchor-verifier-tmp ${CMAKE_CURRENT_SOURCE_DIR}/anchoring/policy $<TARGET_FILE:pchie_anchor>)

# Little Nemo's episodic memory ledger (NEML), header only under include/pchie/neml. Recording is on by default in
# the maze drivers; every test below writes its receipts to a folder in the build tree, never the user's ledger.
add_executable(pchie_neml tools/pchie_neml.cpp)
target_link_libraries(pchie_neml PRIVATE pchie_core)
add_executable(pchie_neml_test tests/pchie_neml_test.cpp)
target_link_libraries(pchie_neml_test PRIVATE pchie_core)
add_test(NAME pchie_neml COMMAND pchie_neml_test)
get_property(_pchie_all_tests DIRECTORY PROPERTY TESTS)
foreach(_t IN LISTS _pchie_all_tests)
  set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT "PCHIE_NEML_DIR=${CMAKE_CURRENT_BINARY_DIR}/neml-test")
endforeach()
