# Checksums

Each GitHub Release of Bolt PCHIE carries a `SHA256SUMS` file next to its downloads. The file lists every archive in the release and every program inside the archives. The repository itself does not keep a checksum file, because a file in the tree cannot list the archive of the tree it is in.

Hashes in `SHA256SUMS` are SHA-256, written as lowercase hex, two spaces, then the file name, one line per file. Comments start with `#`.

## Windows

PowerShell:

```
Get-FileHash .\bolt-pchie-v0.1.0-windows-x64.zip -Algorithm SHA256
```

The `Hash` field should match the line for that file in `SHA256SUMS`. You can also check every file listed, from the folder that holds them:

```
Get-Content .\SHA256SUMS | Where-Object { $_ -and $_ -notmatch '^\s*#' } | ForEach-Object {
  $hash, $name = $_ -split '\s+', 2
  $actual = (Get-FileHash -LiteralPath $name -Algorithm SHA256).Hash.ToLower()
  '{0}  {1}  {2}' -f $name, $actual, ($(if ($actual -eq $hash.ToLower()) { 'OK' } else { 'MISMATCH' }))
}
```

Command Prompt, with `certutil`:

```
certutil -hashfile bolt-pchie-v0.1.0-windows-x64.zip SHA256
```

`certutil` prints the digest on its own line, sometimes with spaces between bytes. Ignore the spaces and compare the hex, case-insensitive, with the `SHA256SUMS` line.

## Linux and macOS

```
sha256sum -c --ignore-missing SHA256SUMS
```

To check one file:

```
sha256sum bolt-pchie-v0.1.0-windows-x64.zip
```

The digest must match the corresponding line, case-insensitive.

## Binaries you build yourself

A build from source will not reproduce the release binaries byte for byte, because compilers embed paths and timestamps. What must reproduce is behavior: the quick tests pass, and the Official pins in [Results so far](results.md) come out the same at any thread count.

If a checksum does not match, stop and do not use that file. A moved Official pin in a test is also a finding; that comparison is described in [Results so far](results.md) and in the getting-started pages.
