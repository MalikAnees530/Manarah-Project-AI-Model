# MANARAH ReID: AI-Powered Wedding Guest Photo Retrieval

MANARAH is a professional-grade, high-performance Person Re-Identification (ReID) system designed specifically for event photography. It enables guests to find all photos of themselves within massive datasets (e.g., weddings, conferences) using a single selfie, ensuring 100% privacy and sub-second retrieval.

## Key Features

- **100% Offline Operations:** All processing occurs locally. No data ever leaves the host machine.
- **Zero Cloud APIs:** Uses optimized ONNX and PyTorch models for local inference.
- **Sub-Second Retrieval:** High-performance vector search powered by Qdrant.
- **Privacy-First Design:** Transient image data is never stored; only anonymized vector embeddings persist.

## Quickstart Guide

### 1. Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Place your event photos in the `wedding pictures/` directory.

### 3. Index the Event
Run the initial indexing pass to detect people and generate embeddings:
```bash
python main.py --reindex
```

### 4. Start the API Server
Launch the FastAPI application to enable search and upload endpoints:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Core Assets
- `model_cache/`: Local repository for DINOv2 and InsightFace models.
- `qdrant_data/`: Active vector database storage.
- `titanium_core/`: The vision pipeline processing engine.
- `cluster_registry.json`: Persistent registry for face-to-cluster mapping.

## Project Architecture & Data Flow

MANARAH follows a functional, modular architecture designed for high-throughput event processing:

- **Step 1: Data Ingestion (`/wedding pictures` & `/uploads`)**: Raw images are ingested via the initial `--reindex` CLI command or dynamically via the `/upload` API endpoint.
- **Step 2: AI Inference (`/model_cache` & `yolo11n-pose.pt`)**: The inference engine utilizes local DINOv2 (Vision Transformer) and Buffalo_L (ArcFace) models for morphological and facial feature extraction.
- **Step 3: Vector Storage (`/qdrant_data`)**: Extracted 512-d face embeddings and 768-d body embeddings are indexed within the Qdrant vector engine for sub-second retrieval.
- **Step 4: Dynamic Operations (`/titanium_core` & `cluster_registry.json`)**: The core vision pipeline orchestrates real-time processing, while `cluster_registry.json` maintains thread-safe mapping between vector identities and physical clusters.
- **Step 5: Client Serving (`/static`)**: Frontend assets and processed result images (with optimized crops and bounding boxes) are served via FastAPI's static file mounts.
