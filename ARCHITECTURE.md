# MANARAH V30.3-Production Architecture

This document details the technical implementation of the MANARAH ReID pipeline, focusing on its multi-modal fusion and bias mitigation strategies.

## 1. Detection & Extraction Layer

The system utilizes a dual-inference strategy for initial person discovery:
- **Facial Embeddings:** InsightFace (Buffalo_L / ArcFace R100) generates high-fidelity 512-dimensional embeddings. It is the primary signal for identity verification.
- **Person Detection:** YOLO11n-pose provides real-time person bounding boxes and skeleton keypoints, essential for the sartorial analysis phase.

## 2. Sartorial Collapse Prevention (SCP)

A major challenge in ReID is "clothing bias"—where the model groups different people simply because they wear similar wedding attire (e.g., black tuxedos). 

MANARAH implements a **Torso-Only Cropping** mechanism:
- Uses YOLO11n-pose keypoints (5, 6, 11, 12) to define the anatomical boundaries of the upper body.
- Crops are processed through **DINOv2 vitb14** (768-d), which has been architected to focus on structural and physiological features rather than color-saturated fabric textures.
- This decoupling ensures that identity is tracked via morphology, not wardrobe.

## 3. Adaptive Scoring & IQR Thresholding

Static thresholds are brittle in varied lighting. MANARAH replaces fixed cutoffs with dynamic **Interquartile Range (IQR)** calculation:
- `face_min` and `body_min` thresholds are derived from the distribution of scores in the live search pool.
- This allows the system to remain highly selective in clear conditions while automatically adjusting sensitivity for challenging, low-light environments.

## 4. Reciprocal Rank Fusion (RRF)

To achieve sub-second precision, the system fuses independent face and body vector search results using a weighted RRF algorithm:
- **Face Dominant Weighting:** Face matches are given a 2.5x multiplier in the fusion rank.
- **Fallback Logic:** In cases where faces are obscured, the system relies on the SCP-weighted body embeddings to maintain identity continuity.

## 5. Thread-Safe Cluster Management

Real-time indexing during live events is handled by the `ClusterRegistry` singleton:
- **Atomic Operations:** Ensures that vector injections from `/upload` endpoints do not collide.
- **Real-Time Merging:** New anchors are merged into existing face banks dynamically, allowing the search engine to improve accuracy as more photos of a guest are discovered.
