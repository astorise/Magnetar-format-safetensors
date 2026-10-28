use super::*;

fn build_file(header_json: &str, data: &[u8]) -> Vec<u8> {
    let header_bytes = header_json.as_bytes();
    let mut file = Vec::with_capacity(HEADER_LENGTH_PREFIX_BYTES + header_bytes.len() + data.len());
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(header_bytes);
    file.extend_from_slice(data);
    file
}

#[test]
fn well_formed_multi_tensor_file_parses_expected_metadata() {
    // Two F32 tensors (4 elements = 16 bytes each) laid out back to back,
    // plus a flat string metadata entry.
    let header = r#"{
        "__metadata__": {"format": "pt"},
        "weight.a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "weight.b": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]}
    }"#;
    let data = vec![0u8; 32];
    let file = build_file(header, &data);

    let artifact = parse(&file).expect("well-formed file parses");
    assert_eq!(
        artifact.metadata.get("format").map(String::as_str),
        Some("pt")
    );
    assert_eq!(artifact.tensors.len(), 2);

    let a = artifact
        .tensors
        .iter()
        .find(|t| t.name == "weight.a")
        .unwrap();
    assert_eq!(a.shape, vec![2, 2]);
    assert_eq!(a.storage_dtype, ModelDType::F32);
    assert_eq!(a.offset_bytes, Some(0));
    assert_eq!(a.size_bytes, Some(16));

    let b = artifact
        .tensors
        .iter()
        .find(|t| t.name == "weight.b")
        .unwrap();
    assert_eq!(b.shape, vec![4]);
    assert_eq!(b.offset_bytes, Some(16));
    assert_eq!(b.size_bytes, Some(16));
}

/// `bind-materialized-weight-content-to-model-artifact-digests`: an F32
/// tensor's `ModelTensorMetadata.digest` is a real content digest, not a
/// placeholder -- it verifies against that exact tensor's real bytes as
/// parsed from the file, and correctly rejects tampered bytes.
#[test]
fn f32_tensor_digest_verifies_against_its_real_bytes() {
    let header = r#"{"weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}"#;
    let data: Vec<u8> = (0..16u8).collect();
    let file = build_file(header, &data);

    let artifact = parse(&file).expect("well-formed file parses");
    let tensor = artifact
        .tensors
        .iter()
        .find(|t| t.name == "weight")
        .unwrap();
    let digest = tensor
        .digest
        .as_ref()
        .expect("an F32 tensor should have a computed content digest");

    digest
        .verify_bytes(&data)
        .expect("digest verifies against the tensor's own real bytes");

    let mut tampered = data;
    tampered[0] ^= 0xFF;
    assert!(
        digest.verify_bytes(&tampered).is_err(),
        "digest must reject tampered content"
    );
}

/// Non-`F32` dtypes cannot be materialized into a `HostTensor` without a
/// re-encoding step this crate does not perform, so their content cannot
/// be meaningfully digested here -- `digest: None` is correct, not an
/// oversight.
#[test]
fn non_f32_tensor_has_no_content_digest() {
    let header = r#"{"weight": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}}"#;
    let data = vec![0u8; 4];
    let file = build_file(header, &data);

    let artifact = parse(&file).expect("well-formed file parses");
    assert!(artifact.tensors[0].digest.is_none());
}

#[test]
fn every_standard_dtype_string_round_trips() {
    let cases: &[(&str, ModelDType, u64)] = &[
        ("BOOL", ModelDType::Bool, 1),
        ("U8", ModelDType::U8, 1),
        ("I8", ModelDType::I8, 1),
        ("U16", ModelDType::U16, 2),
        ("I16", ModelDType::I16, 2),
        ("F16", ModelDType::F16, 2),
        ("BF16", ModelDType::Bf16, 2),
        ("U32", ModelDType::U32, 4),
        ("I32", ModelDType::I32, 4),
        ("F32", ModelDType::F32, 4),
        ("U64", ModelDType::U64, 8),
        ("I64", ModelDType::I64, 8),
        ("F64", ModelDType::F64, 8),
    ];
    for (dtype_str, expected_dtype, width) in cases {
        let header = format!(
            r#"{{"only": {{"dtype": "{dtype_str}", "shape": [3], "data_offsets": [0, {}]}}}}"#,
            3 * width
        );
        let data = vec![0u8; (3 * width) as usize];
        let file = build_file(&header, &data);
        let artifact = parse(&file)
            .unwrap_or_else(|error| panic!("dtype {dtype_str} expected to parse, got {error}"));
        assert_eq!(artifact.tensors[0].storage_dtype, *expected_dtype);
    }
}

#[test]
fn unsupported_f8_dtype_is_rejected_structurally() {
    let header = r#"{"t": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}"#;
    let file = build_file(header, &[0u8]);
    assert!(matches!(
        parse(&file),
        Err(SafetensorsError::UnknownDType { .. })
    ));
}

#[test]
fn overlapping_ranges_are_rejected() {
    let header = r#"{
        "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]}
    }"#;
    let file = build_file(header, &[0u8; 6]);
    assert!(matches!(
        parse(&file),
        Err(SafetensorsError::OverlappingRanges { .. })
    ));
}

#[test]
fn out_of_bounds_range_is_rejected() {
    let header = r#"{"a": {"dtype": "U8", "shape": [16], "data_offsets": [0, 16]}}"#;
    let file = build_file(header, &[0u8; 4]);
    assert!(matches!(
        parse(&file),
        Err(SafetensorsError::ByteRangeOutOfBounds { .. })
    ));
}

#[test]
fn shape_byte_size_mismatch_is_rejected() {
    let header = r#"{"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 4]}}"#;
    let file = build_file(header, &[0u8; 4]);
    assert!(matches!(
        parse(&file),
        Err(SafetensorsError::ShapeByteSizeMismatch { .. })
    ));
}

#[test]
fn shape_overflow_is_rejected_not_panicking() {
    let header =
        r#"{"a": {"dtype": "F32", "shape": [18446744073709551615, 2], "data_offsets": [0, 4]}}"#;
    let file = build_file(header, &[0u8; 4]);
    assert!(matches!(
        parse(&file),
        Err(SafetensorsError::ShapeByteSizeOverflow { .. })
    ));
}

#[test]
fn metadata_non_string_value_is_rejected() {
    let header = r#"{"__metadata__": {"format": 1}}"#;
    let file = build_file(header, &[]);
    assert!(matches!(
        parse(&file),
        Err(SafetensorsError::MetadataValueNotString { .. })
    ));
}

#[test]
fn file_shorter_than_length_prefix_is_rejected() {
    assert!(matches!(
        parse(&[0u8; 4]),
        Err(SafetensorsError::FileTooShortForHeaderLength { .. })
    ));
}

#[test]
fn header_length_exceeding_file_is_rejected() {
    let mut file = 1_000_000_u64.to_le_bytes().to_vec();
    file.extend_from_slice(b"{}");
    assert!(matches!(
        parse(&file),
        Err(SafetensorsError::HeaderLengthExceedsFile { .. })
    ));
}

#[test]
fn non_utf8_header_is_rejected() {
    let mut file = 4_u64.to_le_bytes().to_vec();
    file.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC]);
    assert!(matches!(parse(&file), Err(SafetensorsError::HeaderNotUtf8)));
}

#[test]
fn invalid_json_header_is_rejected() {
    let file = build_file("not json", &[]);
    assert!(matches!(
        parse(&file),
        Err(SafetensorsError::HeaderNotValidJson { .. })
    ));
}

#[test]
fn header_that_is_not_a_json_object_is_rejected() {
    let file = build_file("[1, 2, 3]", &[]);
    assert!(matches!(
        parse(&file),
        Err(SafetensorsError::HeaderNotJsonObject)
    ));
}

#[test]
fn empty_file_is_rejected_not_panicking() {
    assert!(parse(&[]).is_err());
}

/// Corpus regression suite (`reach-architecture-freeze-1` Correctif 16 /
/// `safetensors-format`'s "Never Panics On Malformed Input" requirement):
/// every checked-in malformed byte sequence must be rejected with a
/// structured error, never a panic. Runs under plain `cargo test`, no
/// `cargo-fuzz`/nightly requirement -- see `fuzz/` for the offline fuzz
/// target that grows this corpus over time.
#[test]
fn malformed_input_corpus_never_panics() {
    let corpus_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus");
    let mut checked = 0;
    for entry in std::fs::read_dir(&corpus_dir).expect("corpus directory exists") {
        let entry = entry.expect("readable corpus directory entry");
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("bin") {
            continue;
        }
        let bytes = std::fs::read(&path).expect("readable corpus file");
        let result = std::panic::catch_unwind(|| parse(&bytes));
        assert!(
            result.is_ok(),
            "parsing corpus file {path:?} must not panic"
        );
        assert!(
            result.unwrap().is_err(),
            "corpus file {path:?} is expected to be malformed (parse should fail, not succeed)"
        );
        checked += 1;
    }
    assert!(checked > 0, "corpus directory must not be empty");
}

/// Builds real Safetensors bytes for `magnetar-runtime`'s E2E fixture
/// weights (`e2e_fixture_config`/`e2e_fixture_weights`, both public exactly
/// so this crate's tests can call them). Lives here, not in
/// `magnetar-runtime`, because generating *and proving parseable* a real
/// fixture file needs both `serialize` and `parse` in the same test, and
/// `magnetar-runtime` cannot depend on this crate even as a dev-dependency
/// (`materialize-weights-from-real-model-artifact`'s design.md).
fn build_e2e_fixture_safetensors_bytes() -> Vec<u8> {
    let config = magnetar_runtime::e2e_fixture_config();
    let weights = magnetar_runtime::e2e_fixture_weights(&config)
        .expect("deterministic fixture weights build");

    // Owns the raw little-endian byte buffers so `TensorToSerialize` can
    // borrow from them; `weights` is a `BTreeMap`, so iteration order (and
    // therefore each tensor's byte offset in the file) is already
    // deterministic by tensor name.
    let byte_buffers: Vec<(String, Vec<u8>)> = weights
        .iter()
        .map(|(name, tensor)| {
            let mut bytes = Vec::with_capacity(tensor.data.len() * 4);
            for value in &tensor.data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            (name.clone(), bytes)
        })
        .collect();

    let tensors: Vec<TensorToSerialize<'_>> = byte_buffers
        .iter()
        .map(|(name, bytes)| TensorToSerialize {
            name,
            dtype: ModelDType::F32,
            shape: weights[name].shape.clone(),
            data: bytes,
        })
        .collect();

    serialize(&tensors, &BTreeMap::new())
        .expect("fixture tensors are all F32, serialize cannot fail")
}

#[test]
fn e2e_fixture_weights_round_trip_through_real_safetensors_bytes() {
    let config = magnetar_runtime::e2e_fixture_config();
    let expected_weights = magnetar_runtime::e2e_fixture_weights(&config).unwrap();

    let file_bytes = build_e2e_fixture_safetensors_bytes();
    let artifact = parse(&file_bytes).expect("generated fixture bytes parse cleanly");

    assert_eq!(artifact.tensors.len(), expected_weights.len());
    for tensor in &artifact.tensors {
        assert_eq!(tensor.storage_dtype, ModelDType::F32);
        let expected = &expected_weights[&tensor.name];
        assert_eq!(tensor.shape, expected.shape);

        let start = usize::try_from(tensor.offset_bytes.unwrap()).unwrap();
        let end = usize::try_from(tensor.size_bytes.unwrap()).unwrap() + start;
        let data_start = HEADER_LENGTH_PREFIX_BYTES
            + usize::try_from(u64::from_le_bytes(file_bytes[..8].try_into().unwrap())).unwrap();
        let raw = &file_bytes[data_start + start..data_start + end];
        let values: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        assert_eq!(values, expected.data);
    }
}

/// Prints the real fixture file's bytes as a hex string on stdout, and its
/// sha256 digest, when run with `cargo test -- --ignored --nocapture
/// print_e2e_fixture_safetensors_bytes_for_regeneration`. Not part of the
/// normal test run: this is the manual, one-off generation step for the
/// checked-in `magnetar-runtime/fixtures/e2e-fixture-weights.safetensors`
/// file (the same manual two-command pattern the real Qwen wasm Component
/// binary fixture already uses), re-run only when
/// `e2e_fixture_weights`/`e2e_fixture_config` change.
#[test]
#[ignore = "manual fixture-regeneration tool, not a real test assertion"]
fn print_e2e_fixture_safetensors_bytes_for_regeneration() {
    let bytes = build_e2e_fixture_safetensors_bytes();
    let digest = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hasher.finalize()
    };
    println!(
        "sha256:{}",
        digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    for byte in &bytes {
        print!("{byte:02x}");
    }
    println!();
}
