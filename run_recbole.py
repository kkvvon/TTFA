import argparse
from recbole.quick_start import run_recbole

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        "-m", type=str, default="SASRec")
    parser.add_argument("--dataset",      "-d", type=str, required=True)
    parser.add_argument("--config_files",       type=str, default=None)
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(" ") if args.config_files else None

    if args.model == "SASRec_AddInfo":
        from sasrec_addinfo import SASRec_AddInfo
        model = SASRec_AddInfo
    else:
        model = args.model

    run_recbole(model=model, dataset=args.dataset, config_file_list=config_file_list)
