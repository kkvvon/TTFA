"""
Usage:
    python run_eval_ttfa.py --dataset DATASET --save_name SAVE_NAME --n_steps 20 --gpu_id 0
"""
import os, logging, argparse, random
import torch

logging.getLogger("recbole").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

from recbole.quick_start import load_data_and_model
from recbole.evaluator import Evaluator

K_LIST = [5, 10, 20]


def recbole_eval(pos_item_list, topk_matrix, item_num):
    n       = len(pos_item_list)
    pos_t   = torch.tensor(pos_item_list, dtype=torch.long)
    pos_mat = torch.zeros((n, item_num), dtype=torch.int)
    pos_mat[torch.arange(n), pos_t] = 1
    pos_len = pos_mat.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_mat, dim=1, index=topk_matrix)
    cfg = {"metric_decimal_place": 4,
           "metrics": ["Recall", "MRR", "NDCG"],
           "topk": K_LIST}
    return Evaluator(cfg).evaluate({"rec.topk": torch.cat((pos_idx, pos_len), dim=1)})


def get_target_uids(test_uids, seed, n_users):
    uids = list(test_uids)
    random.seed(seed)
    random.shuffle(uids)
    return uids if n_users is None else uids[:n_users]


def evaluate_single_model(model_file, device, seed, n_users):
    config, model, dataset, _, _, test_data = load_data_and_model(model_file)
    model.eval()
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    item_num  = dataset.item_num
    max_k     = max(K_LIST)

    test_lookup = {}
    for batch in test_data:
        inter = batch[0] if isinstance(batch, (list, tuple)) else batch
        for uid, gt in zip(inter[uid_field].tolist(), inter[iid_field].tolist()):
            test_lookup[uid] = gt

    target_set = set(get_target_uids(test_lookup.keys(), seed, n_users))
    pos_list, topk_list = [], []

    with torch.no_grad():
        for batch in test_data:
            inter = batch[0] if isinstance(batch, (list, tuple)) else batch
            uids  = inter[uid_field].tolist()
            mask  = [i for i, u in enumerate(uids) if u in target_set]
            if not mask:
                continue
            sub    = inter[torch.tensor(mask)].to(device)
            scores = torch.nan_to_num(model.full_sort_predict(sub), nan=0.0, posinf=1e4, neginf=-1e4)
            scores[:, 0] = -1e9
            topk = torch.topk(scores, max_k, dim=1).indices.cpu()
            for i, uid in enumerate(uids[j] for j in mask):
                topk_list.append(topk[i].unsqueeze(0))
                pos_list.append(test_lookup[uid])

    return recbole_eval(pos_list, torch.cat(topk_list), item_num)


def evaluate_ttfa(n_steps, device, seed, n_users):
    from infer_ttfa import FEATURE_NAMES, load_model, ttfa_gate, normalize_scores

    _, model0, dataset0, test_data0 = load_model(FEATURE_NAMES[0])
    uid_field  = dataset0.uid_field
    iid_field  = dataset0.iid_field
    item_seq_f = model0.ITEM_SEQ
    item_seq_l = model0.ITEM_SEQ_LEN
    item_num   = dataset0.item_num
    max_k      = max(K_LIST)

    feat_models = {f: load_model(f)[1] for f in FEATURE_NAMES}

    test_lookup = {}
    for batch in test_data0:
        inter = batch[0] if isinstance(batch, (list, tuple)) else batch
        for uid, gt, seq, slen in zip(
            inter[uid_field].tolist(), inter[iid_field].tolist(),
            inter[item_seq_f], inter[item_seq_l].tolist(),
        ):
            test_lookup[uid] = (seq[:slen].tolist(), gt)

    target_uids = get_target_uids(test_lookup.keys(), seed, n_users)
    target_set  = set(target_uids)
    pos_list, topk_list = [], []
    done = 0

    for batch in test_data0:
        inter = batch[0] if isinstance(batch, (list, tuple)) else batch
        uids  = inter[uid_field].tolist()
        mask  = [i for i, u in enumerate(uids) if u in target_set]
        if not mask:
            continue

        sub = inter[torch.tensor(mask)].to(device)
        with torch.no_grad():
            feat_scores = torch.stack([
                torch.nan_to_num(feat_models[f].full_sort_predict(sub), nan=0.0, posinf=1e4, neginf=-1e4)
                for f in FEATURE_NAMES
            ], dim=0).permute(1, 0, 2)

        for b, uid in enumerate(uids[j] for j in mask):
            scores_dict = {f: feat_scores[b, i, :].unsqueeze(0) for i, f in enumerate(FEATURE_NAMES)}
            gate        = ttfa_gate(scores_dict, test_lookup[uid][0], n_steps=n_steps, device=device)
            score_mat   = normalize_scores(
                torch.nan_to_num(torch.stack([scores_dict[f].squeeze(0) for f in FEATURE_NAMES]),
                                 nan=0.0, posinf=1e4, neginf=-1e4)
            )
            fused = (gate.unsqueeze(1) * score_mat).sum(dim=0)
            fused[0] = -1e9
            topk_list.append(torch.topk(fused, max_k).indices.cpu().unsqueeze(0))
            pos_list.append(test_lookup[uid][1])

        done += len(mask)
        if done % 500 < len(mask):
            print(f"    {done}/{len(target_uids)} done", flush=True)

    return recbole_eval(pos_list, torch.cat(topk_list), item_num)


def print_results(all_results, n_steps, n_users):
    metrics = ["recall", "mrr", "ndcg"]
    col_w   = 10
    header  = f"{'Model':<22}" + "".join(
        f"{m.upper()+'@'+str(k):>{col_w}}" for k in K_LIST for m in metrics
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for name, res in all_results.items():
        row = f"{name:<22}" + "".join(
            f"{res.get(f'{m}@{k}', 0.0):>{col_w}.4f}" for k in K_LIST for m in metrics
        )
        print(row)
    print(f"{sep}\n[n_steps={n_steps}  n_users={n_users}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   type=str, default="")
    parser.add_argument("--save_name", type=str, required=True,
                        help="Suffix for ./saved/{{feature}}_{{save_name}}/ directories")
    parser.add_argument("--n_steps",   type=int, default=20)
    parser.add_argument("--n_users",   type=int, default=None)
    parser.add_argument("--seed",      type=int, default=2026)
    parser.add_argument("--gpu_id",    type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import infer_ttfa
    infer_ttfa.setup(args.save_name)
    from infer_ttfa import _find_latest_pth, MODEL_DIRS

    print(f"Evaluation  dataset={args.dataset}  save_name={args.save_name}  "
          f"seed={args.seed}  n_users={args.n_users}", flush=True)

    all_results = {}

    baseline_dir = f"./saved/sasrec_{args.save_name}"
    try:
        path = _find_latest_pth(baseline_dir)
        print("  >> SASRec_baseline ...", flush=True)
        all_results["SASRec_baseline"] = evaluate_single_model(path, device, args.seed, args.n_users)
    except FileNotFoundError:
        print("  [SKIP] SASRec_baseline not found", flush=True)

    for fname, fdir in MODEL_DIRS.items():
        try:
            path = _find_latest_pth(fdir)
        except FileNotFoundError:
            print(f"  [SKIP] {fname} not found", flush=True)
            continue
        print(f"  >> AddInfo_{fname} ...", flush=True)
        all_results[f"AddInfo_{fname}"] = evaluate_single_model(path, device, args.seed, args.n_users)

    print(f"  >> TTFA (steps={args.n_steps}) ...", flush=True)
    all_results["TTFA"] = evaluate_ttfa(args.n_steps, device, args.seed, args.n_users)

    print_results(all_results, args.n_steps, args.n_users)
