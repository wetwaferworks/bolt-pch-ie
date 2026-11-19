/* Bolt PCHIE - NanoQuant block and model reconstruction (Milestone 2).
 *
 * Clean-room C++ from the published method: Chong, Kim, Kim and Choi, "NanoQuant:
 * Efficient Sub-1-Bit Quantization of Large Language Models", ICML 2026,
 * PMLR 306:20335-20364, arXiv:2602.06694, official code at
 * github.com/SamsungLabs/NanoQuant (Apache 2.0). The official repository was
 * read only as a reference for the reconstruction schedule the paper states at a
 * high level (per-block full-precision error absorption, LB-ADMM initialization,
 * STE refinement of the factors, and a final scale-only knowledge-distillation
 * pass). No source from that tree is copied here.
 *
 * A residual MLP block is treated as the unit of work: two linears with a GELU
 * in between, y = x + W2 gelu(W1 x). A tiny stack of those blocks is enough to
 * exercise the published loop and the resume path. A later hook can drive a
 * full decoder once a block forward for that model exists. See
 * docs/ternary/nanoquant.md.
 */
#ifndef PCHIE_NANOQUANT_RECON_H
#define PCHIE_NANOQUANT_RECON_H

#include "pchie/pchie_nanoquant.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { PCHIE_NQ_RECON_NAME = 64 };

typedef enum pchie_nq_recon_status {
    PCHIE_NQ_RECON_OK = 0,
    PCHIE_NQ_RECON_BAD_ARGUMENT = 1,
    PCHIE_NQ_RECON_IO = 2,
    PCHIE_NQ_RECON_NO_MEMORY = 3,
    PCHIE_NQ_RECON_PAUSED = 4,      /* stopped at a safe point; resume will continue */
    PCHIE_NQ_RECON_BUSY = 5,        /* GPU or another job is using the machine */
    PCHIE_NQ_RECON_MISMATCH = 6     /* a checkpoint's SHA-256 did not match */
} pchie_nq_recon_status;

typedef struct pchie_nq_recon_config {
    float bits;
    int admm_outer_iters;
    int admm_inner_iters;
    float admm_reg;
    uint64_t seed;
    int samples;
    int seq;
    int nonfact_epochs;
    int fact_epochs;
    int model_epochs;
    int nonfact_batch;
    int fact_batch;
    float nonfact_lr;
    float fact_binary_lr;
    float fact_scale_lr;
    float model_kd_lr;
    float shrinkage;
    int use_cuda;           /* 0 CPU, 1 auto, 2 required; tests use 0 so a live init job is not disturbed */
    int tune_nonfact;
    int tune_fact;
    int tune_model;
    int stop_after_blocks;  /* 0 = run all; >0 = pause after that many newly finished blocks */
    int pause_if_input_ms;  /* if >0, pause when the user has used the machine this recently */
    int respect_gpu_busy;   /* if non-zero, refuse to start when the GPU is already in use */
} pchie_nq_recon_config;

typedef struct pchie_nq_recon_stats {
    int blocks;
    int blocks_already;
    int blocks_done_this_run;
    int paused;
    int used_cuda;
    double last_block_loss;
    double model_kd_loss;
} pchie_nq_recon_stats;

void pchie_nq_recon_config_init(pchie_nq_recon_config *config);

/* A residual-MLP toy model large enough to test the published loop, small enough
 * that a short run barely touches the GPU. Weights and calibration inputs are
 * filled from `seed`. The teacher is a copy of the same initial weights. */
pchie_nq_recon_status pchie_nq_recon_tiny(const char *out_dir, int blocks, int hidden, int mid, int vocab,
                                          const pchie_nq_recon_config *config, pchie_nq_recon_stats *stats,
                                          char *error, size_t error_capacity);

/* Resume or continue a reconstruction directory produced by pchie_nq_recon_tiny
 * (or a previous run of this function). The original model file, if any, is
 * never written. */
pchie_nq_recon_status pchie_nq_recon_resume(const char *out_dir, const pchie_nq_recon_config *config,
                                            pchie_nq_recon_stats *stats, char *error, size_t error_capacity);

/* Last-input age in milliseconds, or a large number if the query is not available. */
uint32_t pchie_nq_last_input_age_ms(void);

/* 1 if another of our jobs, a Play build, or some other process is already
 * using the GPU. why is optional. */
int pchie_nq_gpu_is_busy(char *why, size_t cap);

#ifdef __cplusplus
}
#endif

#endif
