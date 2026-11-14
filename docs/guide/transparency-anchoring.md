# Transparency anchoring

Transparency anchoring is a side log. It reads finished-run artifacts and never feeds a run. It never upgrades a label. It does not touch a public log or a network: packets for a witness can be written to a local folder, and nothing is sent from there automatically.

The tool is `pchie_anchor`. It builds with the root CMake build. Tests are `pchie_anchor_unit` (unit checks) and `pchie_anchor_verifier` (end-to-end checks with throwaway keys and logs under the build folder, removed on a clean pass). The log directory, witness keys, and signing key are gitignored. Key custody is a human decision.

## What is recorded

An anchor record binds a finished run's chain head, run id, label, and weights hash. For a ratcheted chain file, which the tool recognizes automatically, the record also carries `chain_generation` and `epoch_id`. A secret-salt genesis, if used, stores only the salt commitment, never the salt.

The log is a Merkle tree in the manner of RFC 9162, with signed tree heads and inclusion and consistency proofs. Witness cosignatures follow a Sigsum-style design: a witness signs a tree head after checking a consistency proof, not after reading the records themselves.

Signatures on tree heads use a stateful hash-based scheme modeled on RFC 8554 (LMS/LM-OTS, SHA-256, n=32, w=4, h=10). The implementation is not claimed to be interoperable with RFC 8554.

A one-time leaf is never used twice. Every signer takes an exclusive lock on the key that all processes honor, and reserves the next leaf index durably (a unique temp file, flushed, then an atomic rename) before it releases a signature. A crash after the reservation wastes one leaf; it never hands the same leaf out twice. Once all 1024 leaves are used, the key refuses to sign and its file is left unchanged. Signing keys come from the operating system's random generator, and seeds and key buffers are wiped after use (best effort). See [SECURITY.md](../../SECURITY.md) for the status of this code: it is research-grade and has not been independently reviewed.

Typical emit, once a run has finished:

```
pchie_anchor emit log --run-id ID --label Scratch --weights SHA --chain-file CHAIN --expect-head HEAD --witness-dir witness/cortana-producer --witness-id cortana-producer
```

Typical verify:

```
pchie_anchor verify log --anchor-index i --chain-file F --fingerprint "....." --witness-held STH --witness-pub HEX --witness-id cortana-producer
```

Each check prints PASS or FAIL (inclusion, consistency, fingerprint, chain re-derivation, and witness presence as requested). `--policy` selects a witness policy. A local machine can cosign as a file-only witness with `pchie_anchor cosign-local`. Verify of a secret-salt chain also takes `--genesis-salt-file`.

## Fingerprints

A fingerprint is 60 digits in 12 groups of 5, from two 5200-round iterated SHA-256 halves, in the manner of a safety number.

- v1 records use rule v1 over (version, chain head, run id).
- Ratcheted records use rule v2, which also covers `chain_generation`.

Old v1 fingerprints stay valid. The v2 rule exists so that a record cannot quietly change generation and keep the same fingerprint.

## Witness policies

A policy is a text file that lists witness ids, a threshold, and optional required ids. Public keys live in the log's witness registry, not in the policy file. Two policies are defined in the tree:

**three-way** (`anchoring/policy/three-way.txt`), the defined policy. Witnesses are `jd-local`, `cortana-producer`, and `elfie`. The threshold is 3 of 3.

**bench** (`anchoring/policy/bench.txt`), the current local setup. The same three ids are listed. The threshold is 2, and `jd-local` and `cortana-producer` are required. Elfie is listed but not required, so that role does not need to talk to the machine that holds the log.

Elfie is a defined witness role in both policies. This guide names that role and does not include any live witness data, keys, or received cosignatures. A policy can list any number of witnesses and raise or lower the threshold. The tree does not record a live exchange with the Elfie role.

Outgoing packets in `witness_out/` are signed tree heads and consistency proofs only. They do not include records, chains, run ids, or weights. Incoming cosignatures land in `witness_in/` and are taken in with `pchie_anchor intake`.

In the bench policy, the maintainer's cosigner role is played by a local machine (`jd-local`), and Cortana is the producer witness.

## Pinning the log key

A witness pins the public key of the log it cosigns for. The safest way is to pin it ahead of time:

```
pchie_anchor witness-trust <witness_dir> <log_pub_hex>
```

Without that step, the key is pinned on first contact, from the first packet the witness accepts. After that, a packet signed under any other key is refused with `LOG KEY CHANGED`, and a packet for a different log id is refused with `LOG ID CHANGED`. Pinning on first contact trusts whatever key the witness sees first, so set the pin ahead of time when you can.

Witness ids name files, so they are limited to letters, digits, `.`, `_` and `-`, 1 to 64 characters, with no leading dot or dash and no `..`.

## What anchoring does not do

- It does not feed a run, change a seal, or move a label.
- It does not publish a log.
- It does not replace chain verification or human truth-checking. An anchored Official record is still Official only because a person accepted it; an anchored Scratch record is still Scratch.
- Epoch re-roots belong to the ratcheted chain in the harness. Anchoring verifies them when they appear in a chain file. It never re-roots a v1 chain, and the harness never re-roots a v1 chain either.
