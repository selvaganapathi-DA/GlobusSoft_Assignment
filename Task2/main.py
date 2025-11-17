import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import traceback

from verify_data import load_models, verify_images_bytes

app = FastAPI(title="Face Verification API")


try:
    mtcnn, resnet = load_models(device='cpu')
except Exception as e:

    mtcnn = None
    resnet = None
    load_error = str(e)
else:
    load_error = None

@app.get("/")
async def root():
    return {"message": "Face Verification: POST two images to /verify (file_a, file_b)."}

@app.post("/verify")
async def verify_endpoint(
    file_a: UploadFile = File(..., description="First image file (face)"),
    file_b: UploadFile = File(..., description="Second image file (face)"),
    threshold: float = Query(0.6, description="Euclidean distance threshold (default 0.6). Lower = stricter")
):

    if mtcnn is None or resnet is None:
        raise HTTPException(status_code=500, detail=f"Model load error: {load_error}")

    try:
        bytes_a = await file_a.read()
        bytes_b = await file_b.read()


        result = verify_images_bytes(bytes_a, bytes_b, mtcnn, resnet, threshold=threshold)

        # Standardize the output format expected by clients:
        if result.get('verification_result') == 'insufficient_faces' or result.get('result') == 'insufficient_faces':
            # Some implementations return 'result' key; massage both variants
            return JSONResponse(status_code=200, content={
                "verification_result": "insufficient_faces",
                "similarity_score": 0.0,
                "euclidean_distance": None,
                "bounding_boxes": {
                    "faces_image_a": result.get('faces_image_a') or result.get('faces_imageA') or [],
                    "faces_image_b": result.get('faces_image_b') or result.get('faces_imageB') or []
                },
                "message": result.get('message', 'No faces detected in one or both images.')
            })

        # If your verify_images_bytes returns 'result' (dictionary) or 'verification_result' name, support both:
        verification_result = result.get('verification_result') or result.get('result') or 'different person'
        euclid = result.get('euclidean_distance') or result.get('distance') or None
        similarity = result.get('similarity_score') or result.get('similarity') or None

        # bounding boxes: prefer the keys faces_image_a / faces_image_b, else fallback
        bboxes_a = result.get('faces_image_a') or result.get('faces_imageA') or result.get('faces_image_a_bboxes') or []
        bboxes_b = result.get('faces_image_b') or result.get('faces_imageB') or result.get('faces_image_b_bboxes') or []

        # If similarity not provided but euclidean distance exists, compute linear similarity:
        if similarity is None and euclid is not None:
            # linear map: similarity = max(0, 1 - distance/threshold)
            sim = 1.0 - (float(euclid) / float(threshold))
            if sim < 0: sim = 0.0
            if sim > 1: sim = 1.0
            similarity = round(sim, 4)

        response = {
            "verification_result": verification_result,
            "similarity_score": float(similarity) if similarity is not None else None,
            "euclidean_distance": float(euclid) if euclid is not None else None,
            "verification_threshold_used": threshold,
            "bounding_boxes": {
                "faces_image_a": bboxes_a,
                "faces_image_b": bboxes_b
            },
            "matched_indices": result.get('matched_indices') or result.get('matched') or None
        }
        return JSONResponse(status_code=200, content=response)

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


if __name__ == "__main__":
    # Start with: python -m uvicorn main:app --reload
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

