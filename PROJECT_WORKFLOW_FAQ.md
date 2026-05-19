# PROJECT WORKFLOW & FAQ: Event ReID Engine V30.3-Production

This document provides a comprehensive guide to the Event ReID Engine architecture, workflow, and technical strategies for Lead Engineers and stakeholders.

### 1. Plain-English Architecture Summary
The V30.3-Production architecture is an **offline-first, multi-modal biometric search engine**. Unlike traditional search tools that rely solely on facial recognition, Event ReID Engine utilizes a dual-engine approach. It combines **InsightFace** for high-precision facial features and **DINOv2** (Vision Transformers) for morphological body features. These representations are stored in a high-performance **Qdrant Vector Database**, allowing the system to instantly locate wedding guests across thousands of high-resolution photos without ever needing a cloud connection.

### 2. Step-by-Step Workflow
The system processes data through a structured, linear pipeline:

1.  **Ingestion:** Raw photos are loaded from the `/wedding pictures` directory or received via dynamic API uploads.
2.  **Detection:** **YOLO11n-pose** identifies every person in the frame, drawing bounding boxes and mapping skeleton keypoints (shoulders, hips, etc.).
3.  **Extraction:**
    *   **InsightFace** extracts a 512-dimensional geometric face map.
    *   **DINOv2** extracts a 768-dimensional body map focusing on physiological structure.
4.  **Indexing:** These vectors are tagged with a unique `event_id` and safely locked into the Qdrant database for persistent storage.
5.  **Retrieval:** When a guest uploads a selfie, the server converts it to vectors, compares them using **Reciprocal Rank Fusion (RRF)**, applies the **Business Confidence Curve**, and returns only matches with a confidence score over 85%.

### 3. Core Q&A (For Lead Engineers & Stakeholders)

**Q: How long will it take to search through a 5k or 10k image album?**
*Answer:* **Sub-second (milliseconds).** Because we use the Qdrant Vector DB, we are not scanning images one by one. Qdrant utilizes HNSW (Hierarchical Navigable Small World) algorithms to mathematically locate the nearest neighboring vectors in high-dimensional space. This means search latency remains nearly flat whether the database contains 1,000 or 50,000 identities.

**Q: On what basis is the system detecting and matching the image?**
*Answer:* It relies on a **"Dual-Vector Fusion"** basis. It creates a 512-dimensional geometric map of the face (ArcFace) and a 768-dimensional map of the torso (DINOv2). The system compares the mathematical distance (Cosine Similarity) between the query photo and the database photos to determine identity.

**Q: How does it detect the user's face, especially if it is a side profile?**
*Answer:* The InsightFace (buffalo_l) model is highly robust to varying yaw and pitch angles, mathematically projecting 2D pixels into a 3D embedding space. If a face is completely turned away, obscured, or shadowed, our **Reciprocal Rank Fusion (RRF)** fallback logic automatically shifts the search weight to the DINOv2 body vector to maintain the identity track without a visible face.

**Q: What is the concept regarding "clothing search" and how do we handle everyone wearing similar clothes (like black suits or red dresses)?**
*Answer:* This is solved via our proprietary **"Sartorial Collapse Prevention."** Standard ReID models often become biased toward clothing color (grouping everyone in black suits together). Event ReID Engine uses YOLO skeleton keypoints (Shoulders 5, 6 and Hips 11, 12) to crop the image *exactly* at the torso. This strips away lower garments and forces the DINOv2 AI to match based on physiological shoulder structure and body morphology rather than fabric color.

**Q: How do we prevent false positives (like a lookalike showing up)?**
*Answer:* We implement two layers of protection: **Adaptive IQR Thresholding**, which dynamically tightens scoring requirements based on the image pool's lighting/quality, and a **strict "85% Hard-Drop Filter"** in our Business Logic Curve. Any match that falls below a calculated 85% confidence is instantly discarded before the user ever sees it.

**Q: Is the system safe if 50 guests upload selfies at the exact same time?**
*Answer:* **Yes.** The system is architected for concurrency. The face bank is managed by a `ClusterRegistry` utilizing a `threading.RLock()`. It safely queues concurrent API requests and uses atomic file-writing (writing to a `.tmp` file and then replacing) to ensure the JSON registry can never be corrupted by race conditions.
