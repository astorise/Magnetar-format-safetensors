//! Real Safetensors file parsing, normalized into `magnetar-runtime`'s
//! generic Model Artifact types ([`ModelTensorMetadata`], [`ModelDType`]).
//! No Safetensors-specific type crosses this crate's public boundary for
//! tensor data (`safetensors-format` capability).
//!
//! # Format
//!
//! A Safetensors file is: an 8-byte little-endian `u64` header length `N`,
//! followed by `N` bytes of UTF-8 JSON (a flat object mapping tensor name
//! to `{dtype, shape, data_offsets: [start, end)}`, plus an optional
//! `"__metadata__"` key holding a flat string map), followed by the raw
//! tensor byte data the header's `data_offsets` index into.
//!
//! # Safety discipline
//!
//! Every offset, length, and dimension originates in untrusted file bytes.
//! Every size computation uses checked arithmetic and every `u64 -> usize`
//! narrowing uses `try_from`, never a bare `as` cast that could silently
//! truncate on a 32-bit target such as `wasm32-unknown-unknown` (the same
//! concern `magnetar-runtime`'s `host_tensor_element_count` documents).
//! Declared tensor byte ranges are validated against the actual remaining
//! file length, and overlapping ranges are rejected outright -- matching
//! upstream `safetensors`' own post-CVE hardening -- before any tensor data
//! is read.
//!
//! # Dtype coverage
//!
//! [`ModelDType::parse`] already accepts every standard Safetensors dtype
//! string (`BOOL`, `U8`/`I8`, `U16`/`I16`, `U32`/`I32`, `U64`/`I64`, `F16`,
//! `BF16`, `F32`, `F64` -- case-insensitively). The two 8-bit float dtypes
//! Safetensors also defines (`F8_E4M3`, `F8_E5M2`) have no `ModelDType`
//! equivalent yet and are rejected with [`SafetensorsError::UnknownDType`],
//! not silently mapped to a nearby type.

use magnetar_runtime::model::{ModelDType, ModelTensorMetadata};
use std::collections::BTreeMap;
use std::fmt;

/// Size, in bytes, of the little-endian header-length prefix every
/// Safetensors file begins with.
pub const HEADER_LENGTH_PREFIX_BYTES: usize = 8;

/// The normalized result of parsing a Safetensors file: a tensor inventory
/// plus the file's flat `__metadata__` string map, if present. Deliberately
/// not a full `ModelManifest` -- a bare weight file carries no model
/// identity, architecture, or digest; assembling those from external
/// context is the caller's responsibility, not this parser's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SafetensorsArtifact {
    pub tensors: Vec<ModelTensorMetadata>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SafetensorsError {
    FileTooShortForHeaderLength {
        file_length: usize,
    },
    HeaderLengthExceedsFile {
        declared: u64,
        available: u64,
    },
    HeaderNotUtf8,
    HeaderNotValidJson {
        reason: String,
    },
    HeaderNotJsonObject,
    TensorEntryNotJsonObject {
        name: String,
    },
    MetadataNotJsonObject,
    MetadataValueNotString {
        key: String,
    },
    MissingField {
        name: String,
        field: &'static str,
    },
    UnknownDType {
        name: String,
        dtype: String,
    },
    InvalidShape {
        name: String,
    },
    InvalidDataOffsets {
        name: String,
    },
    ByteRangeOutOfBounds {
        name: String,
        start: u64,
        end: u64,
    },
    ShapeByteSizeOverflow {
        name: String,
    },
    ShapeByteSizeMismatch {
        name: String,
        expected: u64,
        declared: u64,
    },
    OverlappingRanges {
        first: String,
        second: String,
    },
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileTooShortForHeaderLength { file_length } => write!(
                f,
                "file of {file_length} bytes is too short to hold an {HEADER_LENGTH_PREFIX_BYTES}-byte header length prefix"
            ),
            Self::HeaderLengthExceedsFile {
                declared,
                available,
            } => write!(
                f,
                "declared header length {declared} exceeds the {available} bytes available after the length prefix"
            ),
            Self::HeaderNotUtf8 => write!(f, "header is not valid UTF-8"),
            Self::HeaderNotValidJson { reason } => write!(f, "header is not valid JSON: {reason}"),
            Self::HeaderNotJsonObject => write!(f, "header is not a JSON object"),
            Self::TensorEntryNotJsonObject { name } => {
                write!(f, "tensor entry '{name}' is not a JSON object")
            }
            Self::MetadataNotJsonObject => write!(f, "'__metadata__' entry is not a JSON object"),
            Self::MetadataValueNotString { key } => {
                write!(f, "'__metadata__' value for key '{key}' is not a string")
            }
            Self::MissingField { name, field } => {
                write!(f, "tensor '{name}' is missing required field '{field}'")
            }
            Self::UnknownDType { name, dtype } => {
                write!(f, "tensor '{name}' declares unrecognized dtype '{dtype}'")
            }
            Self::InvalidShape { name } => write!(f, "tensor '{name}' has an invalid shape"),
            Self::InvalidDataOffsets { name } => {
                write!(f, "tensor '{name}' has invalid data_offsets")
            }
            Self::ByteRangeOutOfBounds { name, start, end } => write!(
                f,
                "tensor '{name}' byte range [{start}, {end}) is out of bounds"
            ),
            Self::ShapeByteSizeOverflow { name } => {
                write!(f, "tensor '{name}' shape byte-size computation overflowed")
            }
            Self::ShapeByteSizeMismatch {
                name,
                expected,
                declared,
            } => write!(
                f,
                "tensor '{name}' declared byte range is {declared} bytes but its shape/dtype implies {expected}"
            ),
            Self::OverlappingRanges { first, second } => write!(
                f,
                "tensor '{first}' and tensor '{second}' declare overlapping byte ranges"
            ),
        }
    }
}

impl std::error::Error for SafetensorsError {}

/// Parses a complete Safetensors file already held in memory. See the
/// module documentation for the format and safety discipline.
pub fn parse(bytes: &[u8]) -> Result<SafetensorsArtifact, SafetensorsError> {
    if bytes.len() < HEADER_LENGTH_PREFIX_BYTES {
        return Err(SafetensorsError::FileTooShortForHeaderLength {
            file_length: bytes.len(),
        });
    }
    let header_length_bytes: [u8; HEADER_LENGTH_PREFIX_BYTES] = bytes[..HEADER_LENGTH_PREFIX_BYTES]
        .try_into()
        .expect("slice length checked above");
    let header_length = u64::from_le_bytes(header_length_bytes);

    // `bytes.len()` is an already-trusted, already-allocated slice length;
    // widening it to `u64` can never truncate on any real Rust target.
    let available = (bytes.len() - HEADER_LENGTH_PREFIX_BYTES) as u64;
    if header_length > available {
        return Err(SafetensorsError::HeaderLengthExceedsFile {
            declared: header_length,
            available,
        });
    }
    let header_length =
        usize::try_from(header_length).map_err(|_| SafetensorsError::HeaderLengthExceedsFile {
            declared: header_length,
            available,
        })?;

    let header_start = HEADER_LENGTH_PREFIX_BYTES;
    let header_end = header_start + header_length;
    let header_bytes = &bytes[header_start..header_end];
    let header_str =
        std::str::from_utf8(header_bytes).map_err(|_| SafetensorsError::HeaderNotUtf8)?;
    let header_value: serde_json::Value =
        serde_json::from_str(header_str).map_err(|error| SafetensorsError::HeaderNotValidJson {
            reason: error.to_string(),
        })?;
    let header_object = header_value
        .as_object()
        .ok_or(SafetensorsError::HeaderNotJsonObject)?;

    let data_section_length = available - header_length as u64;

    let mut tensors = Vec::with_capacity(header_object.len());
    let mut metadata = BTreeMap::new();
    let mut ranges: Vec<(u64, u64, String)> = Vec::with_capacity(header_object.len());

    for (name, value) in header_object {
        if name == "__metadata__" {
            let object = value
                .as_object()
                .ok_or(SafetensorsError::MetadataNotJsonObject)?;
            for (key, value) in object {
                let value = value
                    .as_str()
                    .ok_or_else(|| SafetensorsError::MetadataValueNotString { key: key.clone() })?;
                metadata.insert(key.clone(), value.to_string());
            }
            continue;
        }

        let entry = value
            .as_object()
            .ok_or_else(|| SafetensorsError::TensorEntryNotJsonObject { name: name.clone() })?;

        let dtype_str = entry
            .get("dtype")
            .and_then(|value| value.as_str())
            .ok_or_else(|| SafetensorsError::MissingField {
                name: name.clone(),
                field: "dtype",
            })?;
        let dtype = ModelDType::parse(dtype_str).ok_or_else(|| SafetensorsError::UnknownDType {
            name: name.clone(),
            dtype: dtype_str.to_string(),
        })?;

        let shape_array = entry
            .get("shape")
            .and_then(|value| value.as_array())
            .ok_or_else(|| SafetensorsError::InvalidShape { name: name.clone() })?;
        let mut shape = Vec::with_capacity(shape_array.len());
        for dimension in shape_array {
            let dimension = dimension
                .as_u64()
                .ok_or_else(|| SafetensorsError::InvalidShape { name: name.clone() })?;
            shape.push(dimension);
        }

        let offsets_array = entry
            .get("data_offsets")
            .and_then(|value| value.as_array())
            .filter(|array| array.len() == 2)
            .ok_or_else(|| SafetensorsError::InvalidDataOffsets { name: name.clone() })?;
        let start = offsets_array[0]
            .as_u64()
            .ok_or_else(|| SafetensorsError::InvalidDataOffsets { name: name.clone() })?;
        let end = offsets_array[1]
            .as_u64()
            .ok_or_else(|| SafetensorsError::InvalidDataOffsets { name: name.clone() })?;
        if start > end || end > data_section_length {
            return Err(SafetensorsError::ByteRangeOutOfBounds {
                name: name.clone(),
                start,
                end,
            });
        }
        let declared_length = end - start;

        let element_count = shape
            .iter()
            .try_fold(1_u64, |count, &dimension| count.checked_mul(dimension))
            .ok_or_else(|| SafetensorsError::ShapeByteSizeOverflow { name: name.clone() })?;
        if let Some(width) = fixed_element_byte_width(dtype) {
            let expected_length = element_count
                .checked_mul(width)
                .ok_or_else(|| SafetensorsError::ShapeByteSizeOverflow { name: name.clone() })?;
            if expected_length != declared_length {
                return Err(SafetensorsError::ShapeByteSizeMismatch {
                    name: name.clone(),
                    expected: expected_length,
                    declared: declared_length,
                });
            }
        }

        ranges.push((start, end, name.clone()));
        tensors.push(ModelTensorMetadata {
            name: name.clone(),
            shape,
            storage_dtype: dtype,
            layout: None,
            shard: None,
            offset_bytes: Some(start),
            size_bytes: Some(declared_length),
            quantization: None,
            expected_compute_dtype: None,
        });
    }

    // `serde_json::Map` iterates in key order only when built without the
    // `preserve_order` feature (a `BTreeMap` under the hood); sorting
    // explicitly here keeps range-overlap checking, and the returned
    // tensor list, deterministic regardless of that feature flag.
    ranges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));
    for pair in ranges.windows(2) {
        let (_, first_end, first_name) = &pair[0];
        let (second_start, _, second_name) = &pair[1];
        if second_start < first_end {
            return Err(SafetensorsError::OverlappingRanges {
                first: first_name.clone(),
                second: second_name.clone(),
            });
        }
    }

    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(SafetensorsArtifact { tensors, metadata })
}

/// Byte width of one element for dtypes with a fixed per-element size.
/// `None` for block-quantized dtypes (`Q4K`/`Q5K`/`Q8`) that `ModelDType`
/// models for GGUF's sake, not Safetensors': no standard Safetensors dtype
/// string maps to them, so a shape/byte-size cross-check is meaningless for
/// them here (bounds/overflow checks on the declared range still apply).
fn fixed_element_byte_width(dtype: ModelDType) -> Option<u64> {
    match dtype {
        ModelDType::Bool | ModelDType::U8 | ModelDType::I8 => Some(1),
        ModelDType::U16 | ModelDType::I16 | ModelDType::F16 | ModelDType::Bf16 => Some(2),
        ModelDType::U32 | ModelDType::I32 | ModelDType::F32 => Some(4),
        ModelDType::U64 | ModelDType::I64 | ModelDType::F64 => Some(8),
        ModelDType::Q4K | ModelDType::Q5K | ModelDType::Q8 => None,
    }
}

#[cfg(test)]
mod tests;
