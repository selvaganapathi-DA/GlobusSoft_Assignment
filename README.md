# GlobusSoft_Assignment
Junior DataScientist 
Task 1: 
Amazon Laptop Data Scraper
This script scrapes product listings from Amazon.in for the search term "laptop".
It extracts:
Product Image URL
Product Title
Rating (with intelligent fallbacks)
Price
Ad / Organic Result
All extracted data is saved into a CSV file with timestamp.
✅ Features
Scrapes real Amazon search results for laptops
Robust parsing method using BeautifulSoup
Handles sponsored ads & organic results
Intelligent extraction of ratings (even hidden variants)
Extracts lazy-loaded (data-src) images correctly
Automatically waits random seconds to reduce blocking
Exports CSV with timestamp in the script directory
Technologies Used
Python 3.x
Requests
BeautifulSoup4
Pandas
LXML
CSV
├── amazon_laptop_scraper.py      # Main scraping script
├── requirements.txt  

--------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                             Task2:
--------------------------------------------------------------------------------------------------------------------------------------------------------
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

## Project Structure

├── main.py # FastAPI application
├── verify_data.py # Model loading + verification logic
├── train_data.py # Optional training script
├── enrolled_db.npz # Optional saved embeddings (if used)
├── requirements.txt


