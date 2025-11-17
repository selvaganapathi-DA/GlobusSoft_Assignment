from typing import Tuple, Dict, Any, List, Optional
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import io

def load_models(device: str = 'cpu'):
  
    mtcnn = MTCNN(keep_all=True, device=device)  # keep_all True to detect all faces
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet

def image_bytes_to_pil(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')

def detect_faces(mtcnn: MTCNN, pil_img: Image.Image):
   
    faces, probs = mtcnn(pil_img, return_prob=True)
    # faces is None if no faces found
    if faces is None:
        return [], [], []
    # faces: tensor shape (N,3,160,160) or list? typically torch.Tensor
    boxes, _ = mtcnn.detect(pil_img)
    # convert boxes to int tuples
    boxes_out = []
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.tolist())
            boxes_out.append((y1, x2, y2, x1))  # convert to (top,right,bottom,left) to match face_recog style
    # probabilities
    probs_out = [float(p) if p is not None else None for p in (probs.tolist() if hasattr(probs, 'tolist') else probs)]
    # convert faces to cpu numpy arrays if needed
    return faces, boxes_out, probs_out

def embedding_from_face_tensor(resnet: InceptionResnetV1, face_tensor: torch.Tensor, device: str = 'cpu'):
   
    with torch.no_grad():
        if face_tensor.dim() == 3:
            batch = face_tensor.unsqueeze(0).to(device)
        else:
            batch = face_tensor.to(device)
        emb = resnet(batch)  # (1,512)
        emb_np = emb.cpu().numpy().reshape(-1)
        emb_np = emb_np / np.linalg.norm(emb_np)
        return emb_np

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def distance_to_similarity(distance: float, threshold: float = 0.6) -> float:
    # linear map for interpretability (clamp)
    sim = 1.0 - (distance / threshold)
    if sim < 0:
        sim = 0.0
    if sim > 1:
        sim = 1.0
    return float(sim)

def verify_embeddings(a_emb: np.ndarray, b_emb: np.ndarray, threshold: float = 0.6) -> Dict[str, Any]:
    d = euclidean_distance(a_emb, b_emb)
    cos = cosine_similarity(a_emb, b_emb)
    sim = distance_to_similarity(d, threshold=threshold)
    result = 'same person' if d < threshold else 'different person'
    return {
        'result': result,
        'euclidean_distance': float(d),
        'cosine_similarity': float(cos),
        'similarity_score': float(sim)
    }

def verify_images_bytes(img_bytes_a: bytes, img_bytes_b: bytes,
                        mtcnn: MTCNN, resnet: InceptionResnetV1,
                        threshold: float = 0.6) -> Dict[str, Any]:
   
    pil_a = image_bytes_to_pil(img_bytes_a)
    pil_b = image_bytes_to_pil(img_bytes_b)

    faces_a, boxes_a, probs_a = detect_faces(mtcnn, pil_a)
    faces_b, boxes_b, probs_b = detect_faces(mtcnn, pil_b)

    # check faces found
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return {
            'verification_result': 'insufficient_faces',
            'message': f'Faces found - image A: {len(boxes_a)}, image B: {len(boxes_b)}',
            'faces_image_a': boxes_a,
            'faces_image_b': boxes_b
        }

    # pick largest face in each (heuristic) if multiple found
    def pick_largest_index(boxes):
        areas = [(b[2]-b[0])*(b[1]-b[3]) for b in boxes]  # (top,right,bottom,left)
        return int(np.argmax(areas))

    idx_a = pick_largest_index(boxes_a)
    idx_b = pick_largest_index(boxes_b)

    # faces_a is tensor (N,3,160,160); select index
    face_tensor_a = faces_a[idx_a]
    face_tensor_b = faces_b[idx_b]

    emb_a = embedding_from_face_tensor(resnet, face_tensor_a)
    emb_b = embedding_from_face_tensor(resnet, face_tensor_b)

    res = verify_embeddings(emb_a, emb_b, threshold=threshold)
    res.update({
        'faces_image_a': boxes_a,
        'faces_image_b': boxes_b,
        'matched_indices': {'image_a_index': idx_a, 'image_b_index': idx_b}
    })
    return res

def load_enrolled_db(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    names = data['names']
    means = data['means']  # shape (P,512)
    counts = data['counts']
    return names.tolist(), means.astype(np.float32), counts.tolist()

def verify_against_db(img_bytes: bytes, mtcnn: MTCNN, resnet: InceptionResnetV1,
                      names: List[str], means: np.ndarray, threshold: float = 0.6):
   
    pil = image_bytes_to_pil(img_bytes)
    faces, boxes, probs = detect_faces(mtcnn, pil)
    if len(boxes) == 0:
        return {'result': 'no_face', 'message': 'No face found in query image'}

    idx = int(np.argmax([ (b[2]-b[0])*(b[1]-b[3]) for b in boxes ]))
    face_tensor = faces[idx]
    emb = embedding_from_face_tensor(resnet, face_tensor)
    results = []
    for i, name in enumerate(names):
        mean_emb = means[i]
        d = float(euclidean_distance(emb, mean_emb))
        sim = distance_to_similarity(d, threshold=threshold)
        results.append({'name': name, 'distance': d, 'similarity': sim})
    # sort by distance ascending
    results = sorted(results, key=lambda x: x['distance'])
    return {
        'query_bboxes': boxes,
        'matched': results
    }

if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgA', required=True, help='Path to image A')
    parser.add_argument('--imgB', required=True, help='Path to image B')
    parser.add_argument('--threshold', type=float, default=0.6)
    args = parser.parse_args()

    mtcnn, resnet = load_models(device='cpu')
    with open(args.imgA,'rb') as f: a_bytes=f.read()
    with open(args.imgB,'rb') as f: b_bytes=f.read()

    out = verify_images_bytes(a_bytes, b_bytes, mtcnn, resnet, threshold=args.threshold)
    import json
    print(json.dumps(out, indent=2))

