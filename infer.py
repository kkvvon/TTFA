import os
import argparse
import torch
import torch.nn as nn
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores

FEATURE_NAMES = ["price", "sales_rank", "review_emb", "desc_emb"]
MODEL_DIRS: dict = {}
_loaded_models: dict = {}


def setup(save_name: str) -> None:
    global MODEL_DIRS, FEATURE_NAMES, _loaded_models
    MODEL_DIRS = {
        "price":      f"./saved/price_{save_name}/",
        "sales_rank": f"./saved/salesrank_{save_name}/",
        "review_emb": f"./saved/review_{save_name}/",
        "desc_emb":   f"./saved/desc_{save_name}/",
    }
    FEATURE_NAMES = list(MODEL_DIRS.keys())
    _loaded_models = {}


def _find_latest_pth(directory: str) -> str:
    import glob
    files = glob.glob(os.path.join(directory, "*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pth file found in: {directory}")
    return max(files, key=os.path.getmtime)


def load_model(feature_name: str):
    if feature_name in _loaded_models:
        return _loaded_models[feature_name]
    model_file = _find_latest_pth(MODEL_DIRS[feature_name])
    print(f"  Loading [{feature_name}] from: {os.path.basename(model_file)}")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    _loaded_models[feature_name] = (config, model, dataset, test_data)
    return config, model, dataset, test_data


def normalize_scores(score_matrix: torch.Tensor) -> torch.Tensor:
    mean = score_matrix.mean(dim=1, keepdim=True)
    std  = score_matrix.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (score_matrix - mean) / std


def get_all_scores(uid_series, device) -> dict:
    scores = {}
    for fname in FEATURE_NAMES:
        _, model, _, test_data = load_model(fname)
        scores[fname] = full_sort_scores(uid_series, model, test_data, device=device)
    return scores


def ttfa_gate(
    all_scores: dict,
    history_iids: list,
    n_steps: int = 20,
    lr: float = 0.05,
    k: int = 3,
    device: torch.device = None,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    n_features = len(FEATURE_NAMES)

    score_matrix = torch.stack(
        [all_scores[f].squeeze(0) for f in FEATURE_NAMES], dim=0
    ).to(device)
    score_matrix = torch.nan_to_num(score_matrix, nan=0.0, posinf=1e4, neginf=-1e4)
    score_matrix = normalize_scores(score_matrix)

    if len(history_iids) == 0:
        return torch.sigmoid(torch.zeros(n_features, device=device))

    n_items  = score_matrix.shape[1]
    k_eff    = min(k, len(history_iids))
    last_k_t = torch.tensor(history_iids[-k_eff:], dtype=torch.long, device=device)

    with torch.no_grad():
        init_logits = torch.stack([score_matrix[i][last_k_t].mean() for i in range(n_features)])

    gate_logits = nn.Parameter(init_logits.clone())
    optimizer   = torch.optim.Adam([gate_logits], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        gate         = torch.sigmoid(gate_logits)
        fused_scores = (gate.unsqueeze(1) * score_matrix).sum(dim=0)
        neg_iids     = torch.randint(1, n_items, (len(last_k_t),), device=device)
        loss = -torch.log(
            torch.sigmoid(fused_scores[last_k_t] - fused_scores[neg_iids]) + 1e-8
        ).mean()
        loss.backward()
        optimizer.step()

    return torch.sigmoid(gate_logits.detach())


def get_user_history_iids(user_token: str, dataset, test_data) -> list:
    uid_field    = dataset.uid_field
    iid_field    = dataset.iid_field
    uid_internal = dataset.token2id(uid_field, [user_token])[0]
    mask         = dataset.inter_feat[uid_field] == uid_internal
    iid_series   = dataset.inter_feat[iid_field][mask].tolist()
    return iid_series[:-1] if len(iid_series) > 1 else iid_series


def recommend(user_token: str, topK: int = 10, n_steps: int = 20, device=None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, dataset0, test_data0 = load_model(FEATURE_NAMES[0])
    uid_series   = dataset0.token2id(dataset0.uid_field, [user_token])
    all_scores   = get_all_scores(uid_series, device)
    history_iids = get_user_history_iids(user_token, dataset0, test_data0)
    gate         = ttfa_gate(all_scores, history_iids, n_steps=n_steps, device=device)
    gate_dict    = {f: gate[i].item() for i, f in enumerate(FEATURE_NAMES)}

    score_matrix = torch.stack(
        [all_scores[f].squeeze(0) for f in FEATURE_NAMES], dim=0
    ).to(device)
    score_matrix = torch.nan_to_num(score_matrix, nan=0.0, posinf=1e4, neginf=-1e4)
    score_matrix = normalize_scores(score_matrix)

    fused = (gate.unsqueeze(1) * score_matrix).sum(dim=0)
    fused[0] = -1e9
    topk_scores, topk_iids = torch.topk(fused, topK)
    topk_tokens = dataset0.id2token(dataset0.iid_field, topk_iids.cpu().numpy())
    return {"gate": gate_dict, "top_items": list(zip(topk_tokens, topk_scores.cpu().tolist()))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--user_id",   type=str, required=True)
    parser.add_argument("--topK",      type=int, default=10)
    parser.add_argument("--n_steps",   type=int, default=20)
    parser.add_argument("--gpu_id",    type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup(args.save_name)

    result = recommend(args.user_id, topK=args.topK, n_steps=args.n_steps, device=device)
    print(f"Gate: { {k: f'{v:.3f}' for k, v in result['gate'].items()} }")
    print(f"\nTop-{args.topK}:")
    for rank, (item, score) in enumerate(result["top_items"], 1):
        print(f"  {rank:2d}. {item}  ({score:.4f})")
