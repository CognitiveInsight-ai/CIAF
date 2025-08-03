# Lazy Capsule Materialization in ZKETF
**Anchoring Transparency and Privacy-by-Design with Dataset-Level Derivation**

**Author:** Denzil J. Greenwood  
**Date:** July 31, 2025  
**Version:** Draft v1.0

## 1. Introduction
This document outlines the architectural evolution of the Zero-Knowledge Encrypted Transparency Framework (ZKETF) to support Lazy Capsule Materialization. The goal is to preserve the immutability, transparency, and cryptographic trust guarantees of the framework while addressing the performance overhead observed in eager capsule generation.

The proposed design shifts cryptographic anchoring from capsule-level key derivation to dataset-level derivation, with on-demand (lazy) capsule proof generation. This approach maintains audit integrity while substantially reducing computational overhead.

## 2. Problem Statement
Initial tests showed that provenance capsule creation introduced significant overhead:

- **Training with ZKETF:** ~179 seconds
- **Baseline training:** ~2.7 seconds
- **Overhead:** 37,070% (≈371× slower)

The overhead originated primarily from the eager creation of thousands of provenance capsules during training. While cryptographically sound, this approach hindered real-world scalability.

## 3. Design Principles
- **Auditability without Overhead:** Ensure every sample can be proven authentic without eagerly creating all capsules.
- **Privacy-by-Design:** No sensitive information leaves the client; all cryptographic proofs derived client-side.
- **Immutability and Anchoring:** Anchor proofs at the dataset level to ensure consistency and tamper detection.
- **On-Demand Materialization:** Capsules are derived only when requested for audit or compliance checks.

## 4. Cryptographic Hierarchy
### 4.1 Key Derivation Flow
```
Passphrase (Model Name)
        │
        ▼
 PBKDF2-HMAC-SHA256
        │
        ▼
     Master Key
        │
        ├──► Dataset Key = HMAC(Master Key, Dataset Hash)
        │
        └──► Merkle Root (built from sample hashes)
                   │
                   ├── Audit On Demand:
                   │       capsule_key = HMAC(Dataset Key, Capsule ID)
                   │       build full capsule + Merkle proof
                   │
                   └── Anchored Integrity (Merkle root in immutable log)
```

## 5. Metadata Structure
### 5.1 Dataset Metadata
```json
{
  "dataset_id": "kaggle_fake_jobs_20250731",
  "file_name": "fake_job_postings.csv",
  "dataset_hash": "3f7ee6285434579b066a5000c044387040adc5ba8ac4cfb367e1dcd8fce72a78",
  "date_range": "2010-2017",
  "license": "CC0: Public Domain",
  "context": "Kaggle dataset of real and fake job postings",
  "total_samples": 17880,
  "audit_tags": ["provenance", "lazy_capsules", "anchored_merkle_root"]
}
```

### 5.2 Model Metadata
```json
{
  "model_name": "FakeJobDetector_v1",
  "version": "1.0.0",
  "algorithm": "RandomForestClassifier",
  "training_snapshot_id": "2382b2d293b6b95a97ee5ec1cbde6536cb4adc9730f04ee901a2886b68aaf487",
  "master_key_salt": "random128nonce",
  "dataset_anchor": "dataset_kaggle_fake_jobs_20250731",
  "training_timestamp": "2025-07-31T19:25:03.393388"
}
```

### 5.3 Capsule Metadata (On Demand)
```json
{
  "capsule_id": "sample_0457",
  "capsule_key": "HMAC(dataset_key, sample_0457)",
  "merkle_proof": [
    "proof_hash_left",
    "proof_hash_right"
  ],
  "verification_result": true,
  "audit_reference": "provenance_1.0.0_2025-07-31T19-25-04_2335185f"
}
```

### 5.4 Inference Metadata
```json
{
  "inference_id": "inference_948271_capsule_kaggle_fake_jobs_v1_20250731",
  "prediction": "FAKE",
  "confidence": 0.506,
  "timestamp": "2025-07-31T19:25:05.103624",
  "receipt_hash": "b8fbda8c9a9fd6ef57799e88a753db8ee227fba2352ea424369",
  "linked_training_snapshot": "2382b2d293b6b95a97ee5ec1cbde6536cb4adc9730f04ee901a2886b68aaf487",
  "verification": true
}
```

## 6. Audit Trail Integration
Every component — dataset, training snapshot, capsule, inference — includes a capsule tag or audit reference that ties back to the dataset anchor. This ensures lineage and accountability across the entire lifecycle:

**Dataset → Model → Capsule → Inference**

Anchoring all derived keys with the model's master key salt ensures immutability and prevents adversarial manipulation.

## 7. Advantages of the Lazy Approach
- **Reduced Overhead:** Capsules generated only when required (no mass upfront cost).
- **Cryptographic Consistency:** Anchors at dataset level ensure proofs cannot drift.
- **Audit Readiness:** All events tied back to dataset + model anchors.
- **Privacy & Security:** Client-side derivation maintains zero-knowledge guarantees.
- **Scalability:** Works with datasets from thousands to millions of samples.

## 8. Conclusion
The adoption of Lazy Capsule Materialization represents a crucial evolution of ZKETF:

- It reduces computational costs while maintaining full transparency and accountability.
- It provides a privacy-by-design architecture aligned with regulatory requirements.
- It future-proofs the framework for large-scale, high-stakes AI deployments.

**Next Step:** Implement the updated derivation scheme in provenance.py with support for PBKDF2 master keys, dataset-level anchors, and lazy capsule proof generation.
