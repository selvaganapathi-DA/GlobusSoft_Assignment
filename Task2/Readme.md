# Face Verification API (FastAPI + FaceNet)

This project implements **Face Authentication (Face Verification)** using:

- **MTCNN** for face detection  
- **FaceNet (InceptionResnetV1)** for 512-dimensional face embeddings  
- **FastAPI** for a production-ready REST API  
- **PyTorch** as backend for inference  

The system supports pairwise face verification:
- Accepts **two images**
- Detects faces
- Extracts embeddings
- Computes similarity
- Returns:
  - `verification_result`: `"same person"` or `"different person"`
  - `similarity_score` (0.0–1.0)
  - `euclidean_distance`
  - `bounding_boxes` for both images

---

## 📁 Project Structure

├── main.py # FastAPI application
├── verify_data.py # Model loading + verification logic
├── train_enrollment.py # Optional enrollment/training script
├── enrolled_db.npz # Optional saved embeddings (if used)
├── requirements.txt
└── README.md



