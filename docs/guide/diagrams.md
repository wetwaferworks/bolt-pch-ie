# Diagrams

These diagrams describe how the pieces in this tree fit together. They are drawn from the code under `include/`, `src/`, `tools/`, `vision/`, and `anchoring/`, and use the class and tool names found there. The [Architecture](architecture.md) page has the overview flowchart and the committed-step sequence; this page adds a component view, a deployment view, a detailed decision step, the key interfaces, and the run lifecycle.

## Components

```mermaid
flowchart LR
  subgraph core [Core harness: pchie_core]
    Model[IModel / NemotronModel<br/>GGUF loader, deterministic kernels, KV pools]
    Search[MctsSearch over PuctSearch]
    Stack[PchStack: HTN planner, NCM candidates]
    Gate[PchGate]
    Drivers[Maze drivers<br/>HeadlessMaze, StageBDoorMaze]
    Chain[IHashChain<br/>Sha256HashChain, RatchetHashChain]
    Dispatch[IJobDispatcher<br/>SerialJobDispatcher, JoinJobDispatcher]
    Model --> Search --> Stack --> Gate --> Drivers --> Chain
    Dispatch -.-> Search
  end
  subgraph rl [Reasoning layers: include/pchie/rl]
    Guard[LoopGuard]
    Leaf[ILeafValueSource<br/>model, null, oracle]
    Memory[IPlanMemory: query-before-plan]
    Tuples[DecisionTuple side chain]
  end
  subgraph mirror [Mirror-Sim action gate]
    MSG[MirrorSimGate]
    Rule[IUnsafeRule / PitTrapRule]
    MSG --> Rule
  end
  subgraph bridge [Engine bridge]
    Pilot[pchie_lk_pilot<br/>LkPilotController]
    Host[ILkPilotHost / LithoKraftEngineHost]
    Mcp[McpStdioClient]
    Game[LithoKraft on LithoLattice]
    Pilot --> Host --> Mcp --> Game
  end
  subgraph vision [Vision path]
    Visnav[pchie_lk_visnav<br/>rendered frames encoded to text]
    Crutch[CrutchAudit]
    Visnav --> Crutch
  end
  subgraph side [Side observers]
    Tesla[StreamAuthSession<br/>TeslaWrappingChain]
    Watcher[pchie_tesla_watcher]
    Attest[AttestationVerdict<br/>known-good registry]
    Anchor[pchie_anchor<br/>Merkle log, STH, fingerprints]
    Witness[Witness cosigners]
    Tesla --> Watcher
    Anchor --> Witness
  end
  Guard -.-> Drivers
  Leaf -.-> Search
  Memory -.-> Search
  Drivers -.-> Tuples
  Drivers --> MSG
  Drivers -.-> Pilot
  Visnav --> Model
  Visnav -.-> Game
  Chain --> Tesla
  Drivers -.-> Attest
  Chain -.->|finished run artifacts| Anchor
```

Solid arrows are the decide path. Dotted arrows are opt-in layers or observers that read a run without feeding it.

## Deployment

```mermaid
flowchart TB
  subgraph ws [Development workstation]
    Build[Release build: harness, tests, tools]
    Runs[Harness runs and quick tests]
    GameProc[LithoKraft game process<br/>MCP over stdio]
    Weights[(Model weights, outside the repository)]
    Log[(Local anchor log and keys, gitignored)]
    TPM[TPM 2.0<br/>PCR 23 only, bench check]
    Build --> Runs
    Weights --> Runs
    Runs <-->|pilot actuation| GameProc
    Runs -->|finished artifacts| Log
    Runs -.->|quote, optional| TPM
  end
  subgraph cb [Compute box, optional]
    LBuild[Linux Release build]
    LTests[Quick tests at every hardware thread]
    LBuild --> LTests
  end
  subgraph wit [Witnesses]
    W1[Producer witness, local]
    W2[Independent witness]
  end
  Src[(Source repository)] --> Build
  Src --> LBuild
  Log -->|signed tree heads and consistency proofs only| W2
  W2 -->|cosignature file| Log
  W1 --- Log
```

The anchor tool never opens a network connection. Packets and cosignatures move as files, and a witness only ever sees signed tree heads and consistency proofs, never records, chains, run ids, or weights. Thread count does not change any hash, so a run on either machine must reproduce the same Official pins.

## One decision step

```mermaid
sequenceDiagram
  participant Drv as Maze driver
  participant Guard as LoopGuard
  participant Srch as MctsSearch / PuctSearch
  participant Mdl as IModel
  participant Gate as PchGate
  participant Mir as MirrorSimGate (opt-in)
  participant Coord as Coordinator
  participant Chain as IHashChain
  participant Tes as StreamAuthSession
  Drv->>Guard: observe state
  Guard-->>Drv: normal, or reasoning check (fresh seed, larger budget)
  Drv->>Srch: search the root with the node budget
  loop each simulation
    Srch->>Mdl: forward on a forked branch (copy-on-write KV)
    Mdl-->>Srch: logits and priors
    Srch->>Gate: proposal tokens
    Gate-->>Srch: pass, or reject (edge dropped, fail closed)
  end
  Srch-->>Drv: visit counts, commit-time pick
  Drv->>Mir: simulate the committed action on a grid copy
  Mir-->>Drv: allow, or refuse an unsafe action
  Drv->>Coord: step record
  Coord->>Chain: append in canonical step order
  Chain->>Tes: on_append(seal, record) after the append succeeds
  Note over Coord,Chain: Worker threads never append. Results fold in work-id order.
```

## Key interfaces

```mermaid
classDiagram
  class IModel {
    <<interface>>
    +load(path, err) bool
    +tokenize(text) tokens
    +new_branch() ModelBranch
    +fork(branch) ModelBranch
    +forward(branch, token)
    +logits()
    +weights_sha256()
  }
  class IHashChain {
    <<interface>>
    +reset()
    +append(ChainRecord)
    +head() bytes
    +length() size
    +restore_head(head, length)
  }
  class IForkSnapshot {
    <<interface>>
    +snapshot(node_id, ...)
    +rollback(node_id, ...) bool
    +find(node_id)
  }
  class IJobDispatcher {
    <<interface>>
    +parallel_for(count, fn)
    +size() int
  }
  class IUnsafeRule {
    <<interface>>
    +name()
    +is_unsafe(pre, action, post) bool
  }
  class ILkPilotHost {
    <<interface>>
    +player_state(state, err) bool
    +get_block(x, y, z, id, err) bool
    +move(forward, strafe, sneak, err) bool
    +action(which, err) bool
  }
  class ITraceGrader {
    <<interface>>
    +grade(TraceProbe) TraceGrade
  }
  class ILeafValueSource {
    <<interface>>
  }
  IModel <|.. NemotronModel
  IHashChain <|.. Sha256HashChain
  IHashChain <|.. Fnv1a64HashChain
  IHashChain <|.. RatchetHashChain
  IHashChain <|.. TeslaWrappingChain
  IForkSnapshot <|.. ForkSnapshotStore
  IJobDispatcher <|.. SerialJobDispatcher
  IJobDispatcher <|.. JoinJobDispatcher
  IUnsafeRule <|.. PitTrapRule
  ILkPilotHost <|.. LithoKraftEngineHost
  ITraceGrader <|.. SimTraceGrader
  ILeafValueSource <|.. ModelLeafValue
  ILeafValueSource <|.. NullLeafValue
  ILeafValueSource <|.. OracleLeafValue
  TeslaWrappingChain o-- IHashChain : wraps the official chain
  MirrorSimGate o-- IUnsafeRule : registered rules
  PchStack --> PchGate
  PchStack --> MctsSearch
  MctsSearch --> IModel
```

`TeslaWrappingChain` forwards every append to the official chain first and only then MACs the seal, so the official head is identical with TESLA on or off.

## Run lifecycle

```mermaid
stateDiagram-v2
  [*] --> Genesis: load weights, set seed and flags
  Genesis --> Deciding: chain at genesis head
  Deciding --> Sealed: step appended to the chain
  Sealed --> Deciding: next step
  Deciding --> Forked: DualPool rollback (fork)
  Forked --> NewEpoch: restore_head, then re-root
  Deciding --> Frozen: watchdog freeze
  Frozen --> NewEpoch: resume from the last good head
  NewEpoch --> Deciding: ratcheted chains open a new epoch; v1 chains continue unchanged
  Deciding --> Finished: exit reached, budget spent, or stall
  Finished --> Closed: TESLA keys disclosed, attestation verdict recorded
  Closed --> Anchored: optional pchie_anchor emit (side log)
  Anchored --> Cosigned: witness cosignatures taken in
  Closed --> [*]
  Cosigned --> [*]
```

An epoch re-root names the event (fork or resume) and where it happened, so a forked or resumed run never verifies as one straight line. Anchoring happens after a run is finished and never changes its label.
