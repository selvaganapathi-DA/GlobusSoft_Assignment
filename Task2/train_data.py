import os
import argparse
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from tqdm import tqdm

def compute_embeddings_for_folder(dataset_path: str, device: str = 'cpu'):
   
    mtcnn = MTCNN(keep_all=False, device=device)  # returns single face crop
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    embeddings_db = {}
    for person in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        embeddings_db[person] = []
        # iterate images
        for fname in sorted(os.listdir(person_dir)):
            fp = os.path.join(person_dir, fname)
            # skip non-images
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                continue
            try:
                img = Image.open(fp).convert('RGB')
            except Exception as e:
                print(f"Skipping {fp}: cannot open ({e})")
                continue
            # detect + crop single face (MTCNN). Returns PIL Image or None
            try:
                crop = mtcnn(img)
            except Exception as e:
                print(f"Error running MTCNN on {fp}: {e}")
                crop = None
            if crop is None:
                print(f"No face detected in {fp}; skipping")
                continue
            # crop is tensor CxHxW - move to device and create batch
            with torch.no_grad():
                emb = resnet(crop.unsqueeze(0).to(device))  # shape (1,512)
                emb_np = emb.cpu().numpy().reshape(-1)  # shape (512,)
            embeddings_db[person].append(emb_np)
        # if no embeddings for person, delete entry
        if len(embeddings_db[person]) == 0:
            print(f"Warning: no faces found for {person}; removing from DB")
            del embeddings_db[person]

    return embeddings_db

def build_mean_embedding_db(embeddings_db: dict):
  
    out = {}
    for person, embs in embeddings_db.items():
        arr = np.stack(embs, axis=0)  # shape (N,512)
        mean = np.mean(arr, axis=0)

        mean = mean / np.linalg.norm(mean)
        out[person] = {'mean': mean.astype(np.float32), 'count': arr.shape[0]}
    return out

def save_npz(DB: dict, out_path: str):
  
    names = []
    means = []
    counts = []
    for name, info in DB.items():
        names.append(name)
        means.append(info['mean'])
        counts.append(info['count'])
    names = np.array(names, dtype=object)
    means = np.stack(means, axis=0).astype(np.float32)  # (P,512)
    counts = np.array(counts, dtype=np.int32)
    np.savez_compressed(out_path, names=names, means=means, counts=counts)
    print(f"Saved enrollment DB to: {out_path} (people: {len(names)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset root (subfolders per person)')
    parser.add_argument('--out', default='enrolled_db.npz', help='Output npz path')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0')
    args = parser.parse_args()

    print("Computing embeddings for dataset:", args.dataset)
    emb_db = compute_embeddings_for_folder(args.dataset, device=args.device)
    if not emb_db:
        print("No embeddings computed. Exiting.")
        return
    mean_db = build_mean_embedding_db(emb_db)
    save_npz(mean_db, args.out)

if __name__ == '__main__':
    main()

