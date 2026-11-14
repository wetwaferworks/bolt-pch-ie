# Known-good registry

Append-only. Only the maintainer adds an entry. Cortana cosigns. Existing entries are never edited or removed.

Each line in `known-good-registry.jsonl` is one entry. A valid fresh quote whose measurements match an entry appraises as Attested. Without a quote the verdict is Unattested, even if the measurement is in this list.

A match covers every measured field, the run's TESLA commitment included (see [SECURITY.md](../../SECURITY.md)). Since every run now draws a fresh TESLA secret, an entry matches only the run whose commitment it records. The first entry was recorded on 13 November 2026, before that change, so its commitment came from the old seed derived from the maze seed. It stays as recorded; entries are never edited.
