# safetensors-format Specification

## Purpose
TBD - created by archiving change implement-model-format-parsers. Update Purpose after archive.
## Requirements
### Requirement: Safetensors Parser Produces Only Generic Types
The Safetensors parser SHALL normalize a parsed file into `magnetar-runtime`'s existing generic types (`ModelTensorMetadata`, `ModelDType`) and a flat string metadata map, and SHALL NOT expose any Safetensors-specific type across its public API boundary.

#### Scenario: Tensor inventory normalized
- **WHEN** a well-formed Safetensors file is parsed
- **THEN** the result is a list of `ModelTensorMetadata` values (name, shape, storage dtype, byte offset, byte size) and a `BTreeMap<String, String>` for the file's `__metadata__` section, with no Safetensors-specific struct in the returned type.

### Requirement: Safetensors Parser Rejects Overflow And Out-Of-Bounds Ranges
The parser SHALL use checked arithmetic for every size and offset computation derived from file bytes, and SHALL reject a tensor whose declared byte range is out of bounds, overlapping with another tensor's range, or would overflow during size computation, before any data is read from that range.

#### Scenario: Declared range exceeds file length
- **WHEN** a tensor's declared `[start, end)` byte range extends past the end of the file
- **THEN** the parser returns a structured error and reads no tensor bytes.

#### Scenario: Overlapping tensor ranges
- **WHEN** two tensors declare byte ranges that overlap
- **THEN** the parser returns a structured error rather than accepting either tensor.

#### Scenario: Shape causes overflow
- **WHEN** a tensor's declared shape and dtype combination would overflow `u64` byte-size arithmetic
- **THEN** the parser returns a structured error instead of wrapping or panicking.

### Requirement: Safetensors Parser Never Panics On Malformed Input
The parser SHALL return a structured error rather than panic for any malformed input, including truncated headers, invalid UTF-8, invalid JSON, unknown dtype strings, and non-JSON-object headers.

#### Scenario: Corpus regression suite
- **WHEN** the checked-in malformed-input corpus is replayed through the parser
- **THEN** every entry returns a structured error and none panics.

#### Scenario: Fuzz target exists
- **WHEN** the crate's fuzz target is run against arbitrary byte input
- **THEN** it exercises the same public parse entry point the corpus regression suite uses.

### Requirement: Magnetar-Runtime Has No Safetensors Dependency
`magnetar-runtime` SHALL NOT depend on this crate or any Safetensors-specific type.

#### Scenario: Static dependency guard
- **WHEN** `magnetar-runtime`'s dependency graph is inspected
- **THEN** the Safetensors format crate does not appear in it.

