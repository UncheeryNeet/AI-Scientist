import sys
import os

import numpy as np
import torch


def produce_evaluation_file(
    data_loader,
    model,
    device,
    save_path: str,
    trial_path: str,
) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(" ")
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def calculate_tDCF_EER(cm_scores_file, asv_score_file, output_file, printout=True):

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        "Pspoof": Pspoof,  # Prior probability of a spoofing attack
        "Ptar": (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        "Pnon": (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        "Cmiss": 1,  # Cost of ASV system falsely rejecting target speaker
        "Cfa": 10,  # Cost of ASV system falsely accepting nontarget speaker
        "Cmiss_asv": 1,  # Cost of ASV system falsely rejecting target speaker
        "Cfa_asv": 10,  # Cost of ASV system falsely accepting nontarget speaker
        "Cmiss_cm": 1,  # Cost of CM system falsely rejecting target speaker
        "Cfa_cm": 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    # asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = np.double(asv_data[:, 2])  # .astype(np.float32)

    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    # cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = np.double(cm_data[:, 3])  # astype(np.float32)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == "target"]
    non_asv = asv_scores[asv_keys == "nontarget"]
    spoof_asv = asv_scores[asv_keys == "spoof"]

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == "bonafide"]
    spoof_cm = cm_scores[cm_keys == "spoof"]

    # EERs of the standalone systems and fix ASV operating point to
    # EER threshold
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]

    attack_types = [f"A{_id:02d}" for _id in range(7, 20)]
    if printout:
        spoof_cm_breakdown = {
            attack_type: cm_scores[cm_sources == attack_type]
            for attack_type in attack_types
        }

        eer_cm_breakdown = {
            attack_type: compute_eer(bona_cm, spoof_cm_breakdown[attack_type])[0]
            for attack_type in attack_types
        }

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )

    # Compute t-DCF
    tDCF_curve, CM_thresholds = compute_tDCF(
        bona_cm,
        spoof_cm,
        Pfa_asv,
        Pmiss_asv,
        Pmiss_spoof_asv,
        cost_model,
        print_cost=False,
    )

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write("\nCM SYSTEM\n")
            f_res.write(
                "\tEER\t\t= {:8.9f} % "
                "(Equal error rate for countermeasure)\n".format(eer_cm * 100)
            )

            f_res.write("\nTANDEM\n")
            f_res.write("\tmin-tDCF\t\t= {:8.9f}\n".format(min_tDCF))

            f_res.write("\nBREAKDOWN CM SYSTEM\n")
            for attack_type in attack_types:
                _eer = eer_cm_breakdown[attack_type] * 100
                f_res.write(
                    f"\tEER {attack_type}\t\t= {_eer:8.9f} % (Equal error rate for {attack_type}\n"
                )
        os.system(f"cat {output_file}")

    return eer_cm * 100, min_tDCF


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    # false rejection rates
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Returns equal error rate (EER) and the corresponding threshold."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(
    bonafide_score_cm,
    spoof_score_cm,
    Pfa_asv,
    Pmiss_asv,
    Pmiss_spoof_asv,
    cost_model,
    print_cost,
):
    # Sanity check of cost parameters
    if (
        cost_model["Cfa_asv"] < 0
        or cost_model["Cmiss_asv"] < 0
        or cost_model["Cfa_cm"] < 0
        or cost_model["Cmiss_cm"] < 0
    ):
        print("WARNING: Usually the cost values should be positive!")

    if (
        cost_model["Ptar"] < 0
        or cost_model["Pnon"] < 0
        or cost_model["Pspoof"] < 0
        or np.abs(cost_model["Ptar"] + cost_model["Pnon"] + cost_model["Pspoof"] - 1)
        > 1e-10
    ):
        sys.exit(
            "ERROR: Your prior probabilities should be positive and sum up to one."
        )

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit(
            "ERROR: you should provide miss rate of spoof tests against your ASV system."
        )

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit("ERROR: Your scores contain nan or inf.")

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit("ERROR: You should provide soft CM scores - not binary decisions")

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm
    )

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = (
        cost_model["Ptar"]
        * (cost_model["Cmiss_cm"] - cost_model["Cmiss_asv"] * Pmiss_asv)
        - cost_model["Pnon"] * cost_model["Cfa_asv"] * Pfa_asv
    )
    C2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            "You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?"
        )

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print(
            "t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n".format(
                bonafide_score_cm.size, spoof_score_cm.size
            )
        )
        print("t-DCF MODEL")
        print(
            "   Ptar         = {:8.5f} (Prior probability of target user)".format(
                cost_model["Ptar"]
            )
        )
        print(
            "   Pnon         = {:8.5f} (Prior probability of nontarget user)".format(
                cost_model["Pnon"]
            )
        )
        print(
            "   Pspoof       = {:8.5f} (Prior probability of spoofing attack)".format(
                cost_model["Pspoof"]
            )
        )
        print(
            "   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)".format(
                cost_model["Cfa_asv"]
            )
        )
        print(
            "   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)".format(
                cost_model["Cmiss_asv"]
            )
        )
        print(
            "   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)".format(
                cost_model["Cfa_cm"]
            )
        )
        print(
            "   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)".format(
                cost_model["Cmiss_cm"]
            )
        )
        print(
            "\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)"
        )

        if C2 == np.minimum(C1, C2):
            print(
                "   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n".format(C1 / C2)
            )
        else:
            print(
                "   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n".format(C2 / C1)
            )

    return tDCF_norm, CM_thresholds
