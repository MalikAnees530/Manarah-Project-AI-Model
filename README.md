# 🛡️ MANARAH ReID: AI-Powered Event Guest Photo Retrieval

🤖 **MANARAH** is an enterprise-grade, smartphone-optimized, offline-first Person Re-Identification (ReID) biometric orchestration engine designed, architected, and built exclusively as an independent software project.

[![Engine CI Status](https://github.com/MalikAnees530/Manarah-Project-AI-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/MalikAnees530/Manarah-Project-AI-Model/actions)
[![License Profile](https://img.shields.io/badge/License-MIT-gold.svg)](LICENSE)
[![Ownership Profile](https://img.shields.io/badge/Ownership-Sole%20Independent%20Project-purple.svg)](#)
[![Language Backbone](https://img.shields.io/badge/Backbone-Python%203.10%2B-darkgreen.svg)](requirements.txt)
[![Privacy Shield](https://img.shields.io/badge/Privacy-100%25%20Offline%20%7C%20Zero%20Cloud-red.svg)](ARCHITECTURE.md)

MANARAH operates as a zero-shot standalone AI microservice, eliminating the need for ongoing model retraining loops or cloud processing. It enables guests to instantly locate all photos of themselves across massive event datasets using a single query selfie while ensuring absolute data isolation and sub-second execution speeds.

---

## 💎 Key Structural Features

* 🔒 **100% Offline Infrastructure:** All model inference and calculations occur locally on the host server GPU. No biometric data ever leaves your hardware perimeter.
* 👔 **Sartorial Collapse Prevention (SCP):** Uses anatomical skeleton keypoint coordinates mapped via CNN pose estimations to crop strictly at the upper torso. This isolates body morphology, forcing the transformer to evaluate frame structural metrics rather than clothing textures or color values—crucial for tracking identical uniform attire (e.g., traditional Saudi Arabian wedding Thobes).
* 🧬 **Dual-Vector Rank Fusion (RRF):** Blends high-fidelity 512-dimensional facial coordinate layouts (ArcFace) with dense 768-dimensional body structure maps (ViT) using a weighted rank fusion algorithm. Features a 2.5x prioritization weight on facial traits with automated fallback routing to body vectors if a profile is heavily turned or obscured.
* ⚡ **Sub-Second HNSW Graph Traversal:** Features high-speed proximity queries powered by an integrated local Qdrant Vector Database instance. Bypasses file system overheads by searching structured Hierarchical Navigable Small World graphs, locking search latencies flat to 5-20 milliseconds across tens of thousands of identities.

---

## 🛠️ Technology Stack Architecture

* **Language Environment:** Python 3.10+ (Monolithic ASGI Pipeline)
* **API Framework:** FastAPI with Concurrent Async CORS Workers
* **Vector Engine:** Qdrant DB Engine (Local Multi-Tenant Storage Configuration)
* **Object Discovery:** YOLO11n-pose (Skeletal Landmark Positioning)
* **Biometric Mapping:** InsightFace Buffalo_L (512-d Facial Feature Extraction)
* **Morphology Transformer:** Meta DINOv2 vitb14 (768-d Torso Feature Extraction)

---

## 👤 Author & Creator

* 👨‍💻 **Malik Anees Ahmed** — *Sole Creator & Lead AI Engineer*
  * **Core Focus:** End-to-end mathematical modeling, deep learning pipelines optimization, multi-modal feature vector fusion engineering, and localized database graph orchestration.
  * **Project Type:** 100% Independent Proprietary Development Core.

---

## 📄 License Governance
This system core is independent software proprietary property licensed under the terms of the official [MIT License](LICENSE).
