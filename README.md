# magnetar-format-safetensors

## Purpose

A Safetensors Model Artifact parser for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It reads
Safetensors-encoded weight files and produces Magnetar's generic,
Provider-agnostic types (`ModelTensorMetadata`, `ModelDType`) at the
boundary into `magnetar-runtime` -- `magnetar-runtime` SHALL have zero
dependency on this crate or any Safetensors-specific type, per Magnetar's
architectural invariant that Providers and Format parsers own
native/format-specific implementation details while the Core stays
generic.

## Status

**Real parser, not a template.** `parse(bytes: &[u8]) -> Result<SafetensorsArtifact, SafetensorsError>`
reads the 8-byte header-length prefix, the JSON header, and normalizes every
tensor entry into `ModelTensorMetadata` plus a flat `__metadata__` string
map. Every standard Safetensors dtype string round-trips through
`ModelDType::parse` (`BOOL`, `U8`/`I8`, `U16`/`I16`, `U32`/`I32`, `U64`/`I64`,
`F16`, `BF16`, `F32`, `F64`); the two 8-bit float dtypes Safetensors also
defines (`F8_E4M3`, `F8_E5M2`) have no `ModelDType` equivalent yet and are
rejected structurally, not silently mapped.

Overflow and bounds safety: every offset/size computation uses checked
arithmetic, declared tensor byte ranges are validated against the file's
actual length, and overlapping tensor ranges are rejected outright
(matching upstream `safetensors`' own post-CVE hardening). 15 unit tests
plus a 20-entry checked-in malformed-input corpus (`tests/corpus/`) prove
every category of malformed input is rejected with a structured error, not
a panic. A `cargo-fuzz` target (`fuzz/fuzz_targets/parse.rs`) exercises the
same entry point for offline/periodic fuzzing (`cargo +nightly fuzz build`
verified to build; live execution requires a sanitizer-runtime-equipped
environment -- not exercised on the Windows machine this crate was
developed on, a real, documented gap rather than an implicit claim of full
local fuzzing coverage).

Deliberately not a full `ModelManifest`: a bare Safetensors file carries no
model identity, architecture, or digest, so `SafetensorsArtifact` is a
narrower tensor inventory + metadata map; assembling a `ModelManifest` from
it plus external context is the caller's job.

## Governing contract

[`safetensors-format`](https://github.com/astorise/Magnetar/blob/main/openspec/changes/implement-model-format-parsers/specs/safetensors-format/spec.md)
in the main Magnetar repository's OpenSpec change set defines this crate's
requirements in full (type-boundary discipline, overflow/panic safety,
corpus/fuzz coverage). `model-format-roadmap`'s "Safetensors Support"
requirement records that support as implemented, not aspirational, as of
this crate.

## Relationship to magnetar-runtime

This crate depends only on `magnetar-runtime`'s public `model` module types
(`formats/safetensors -> magnetar-runtime`, never the reverse) --
`magnetar-runtime` compiling and testing cleanly without this crate present,
and never appearing in its own dependency graph, is verified by a static CI
guard (`.github/workflows/quality.yml`'s `submodule-integration` job). It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `formats/safetensors`.
