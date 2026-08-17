import numpy as np

def linear_cka(X, Y):
    """Standard linear CKA (Kornblith et al. 2019), feature-space formulation —
    efficient when #channels << #patches, which is the case here."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Y.T @ X, ord='fro') ** 2
    norm_x = np.linalg.norm(X.T @ X, ord='fro')
    norm_y = np.linalg.norm(Y.T @ Y, ord='fro')
    return cross / (norm_x * norm_y)

def compare_checkpoints(base_npz_path, cfe_npz_path):
    base = np.load(base_npz_path)
    cfe = np.load(cfe_npz_path)

    # align by patch key, in case ordering differs between runs
    base_keys = {tuple(k): i for i, k in enumerate(base["keys"])}
    cfe_keys = {tuple(k): i for i, k in enumerate(cfe["keys"])}
    shared = sorted(set(base_keys) & set(cfe_keys))
    print(f"{len(shared)} matched patches "
          f"(base had {len(base_keys)}, cfe had {len(cfe_keys)})")

    base_idx = [base_keys[k] for k in shared]
    cfe_idx = [cfe_keys[k] for k in shared]

    for stage in ["e1", "e2", "e3", "e4"]:
        X = base[stage][base_idx]
        Y = cfe[stage][cfe_idx]
        cka = linear_cka(X, Y)
        print(f"{stage}: CKA = {cka:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_npz", type=str, required=True)
    parser.add_argument("--cfe_npz", type=str, required=True)
    args = parser.parse_args()
    compare_checkpoints(args.base_npz, args.cfe_npz)