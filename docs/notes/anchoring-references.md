# Anchoring references: what we take and what we leave

Short reading notes for the anchoring and ratchet work. These are reading references only. Everything is implemented in-house on the tree's own SHA-256, with no imports.
How to say it: "a hash-chained verified harness, its seals ratcheted in the manner of Signal's Double Ratchet specification, and its chain heads anchored in an append-only transparency log in the manner of RFC 9162, Certificate Transparency v2, with witnessed signed tree heads after the Sigsum design."

Signal's Double Ratchet is a Signal specification, not an IETF RFC. RFC 9162 covers the transparency-log side only.

## Signal, The Double Ratchet Algorithm (revision 4, 2025-11-04)
https://signal.org/docs/specifications/doubleratchet/
- Take: the KDF chain (s2.1) and the symmetric-key ratchet (s2.2): each step derives a one-time key and replaces the chain key, so earlier keys cannot be computed from later ones. Also KDF_CK as HMAC-SHA256 with separate constant inputs, 0x01 for the message key and 0x02 for the next chain key (s7.2), and HKDF with the previous value as salt for re-keying (KDF_RK, s7.2). We use the same shape for epoch re-roots. And from s8.1, deleting old state matters, so old chain keys and step keys are wiped.
- Leave: the Diffie-Hellman ratchet, AEAD encryption, header encryption, and the post-quantum SPQR and Triple Ratchet (s3-s6). We seal records; we don't encrypt messages. There is also no break-in recovery: with no fresh secret input, a leaked state exposes future steps but not past ones.

## RFC 9162, Certificate Transparency Version 2.0
https://www.rfc-editor.org/rfc/rfc9162
- Take: the Merkle Tree Hash with 0x00 leaf and 0x01 node prefixes (s2.1.1), inclusion proof generation and verification (s2.1.3), consistency proof generation and verification (s2.1.4), and a signed tree head over tree size, root, and timestamp.
- Leave: X.509, precertificates, SCTs, the TLS and HTTP APIs, and the specific signature algorithms. There is no elliptic-curve code in our tree, so tree heads use our hash-based signature instead.

## Sigsum witness cosigning, now C2SP tlog-witness and tlog-cosignature
https://www.sigsum.org/docs/ , https://c2sp.org/tlog-witness , https://c2sp.org/tlog-cosignature
- Take: a witness keeps only the latest head it cosigned for each log, checks a consistency proof from that head before it cosigns, and never cosigns a smaller tree. Cosignatures are monotonic and name the witness, log, size, and root.
- Leave: the HTTP protocol, the signed-note text format, ML-DSA and Ed25519 keys, and public witness networks. Our exchange is files only, and nothing has been sent to anyone.

## Signal key transparency (Automatic Key Verification, August 2026)
https://signal.org/blog/automatic-key-verification/ , https://github.com/signalapp/key-transparency-server
- Take: clients accept a tree head only if independent auditors have endorsed it, which bounds how long a split view can last. That is the same job our witness-held head does.
- Leave: prefix trees, VRF-obscured identifiers, and the privacy machinery. We log run anchors, not user keys.

## Signal safety numbers (fingerprint)
libsignal numeric fingerprint: https://github.com/signalapp/libsignal
- Take: an iterated SHA-256 (5200 rounds), 60 decimal digits read aloud in groups of 5.
- Leave: combining two parties' halves. Ours is one chain head plus one run id. The length is an open question for the maintainer.

## Also used (not part of the brief's lineage)
- RFC 2104 (HMAC) and RFC 5869 (HKDF), with the RFC 4231 and RFC 5869 test vectors checked in `anchor_unit`.
- RFC 8554 (LMS) as the model for the tree-head signature. We do not claim interoperability.
