import json
import os.path as osp
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Define the datasets you're working with
datasets = ["ASVspoof2019"]  # Update as per your datasets

INFO = {"ASVspoof2019": {"size": 25380}}

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baseline",
    "run_1": "Preemphasis Enabled (coef=0.97)",
    "run_2": "Preemphasis Enabled (coef=0.95)",
    "run_3": "Preemphasis Reverted (coef=0.97)",
    "run_4": "Preemphasis Disabled",
    "run_5": "Preemphasis Dynamic Coefficient",
}


# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Initialize dictionaries to store data
folders = os.listdir("./")
final_results = {}
results_info = {}
for folder in folders:
    # Load final_info.json
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        # Load all_results.npy
        results_dict = np.load(
            osp.join(folder, "all_results.npy"), allow_pickle=True
        ).item()
        run_info = {}
        for dataset in datasets:
            # Initialize lists to collect data across seeds
            all_train_losses = []
            all_val_eers = []
            all_val_losses = []

            keys = [k for k in results_dict.keys()]
            train_key = f"{dataset}_0_train_info"
            val_key = f"{dataset}_0_val_info"
            if train_key in results_dict and val_key in results_dict:
                train_info = results_dict[train_key]
                val_info = results_dict[val_key]

                # Extract training data
                train_iters = [
                    entry["epoch"] * final_results[folder][dataset]["batch"]
                    + entry["iter"]
                    for entry in train_info
                ]
                train_losses = [
                    entry["loss"].cpu().detach().numpy() for entry in train_info
                ]
                all_train_losses.append(train_losses)

                # Extract validation data
                val_iters = [entry["epoch"] for entry in val_info]
                val_losses = [entry["loss"] for entry in val_info]
                val_eers = [entry["eer"] for entry in val_info]
                all_val_eers.append(val_eers)
                all_val_losses.append(val_losses)

            # Now compute mean and standard error across seeds
            if all_train_losses:
                train_losses_array = np.array(all_train_losses)
                mean_train_losses = np.mean(train_losses_array, axis=0)
                stderr_train_losses = np.std(train_losses_array, axis=0) / np.sqrt(
                    len(train_losses_array)
                )
            else:
                train_iters_common = []
                mean_train_losses = []
                stderr_train_losses = []

            if all_val_losses and all_val_eers:
                val_losses_array = np.array(all_val_losses)
                mean_val_losses = np.mean(val_losses_array, axis=0)
                stderr_val_losses = np.std(val_losses_array, axis=0) / np.sqrt(
                    len(val_losses_array)
                )
                val_eers_array = np.array(all_val_eers)
                mean_val_eers = np.mean(val_eers_array, axis=0)
                stderr_val_eers = np.std(val_eers_array, axis=0) / np.sqrt(
                    len(val_eers_array)
                )
            else:
                val_iters_common = []
                mean_val_losses = []
                stderr_val_losses = []

            # Store in run_info
            run_info = {
                "train_iters": train_iters,
                "mean_train_losses": mean_train_losses,
                "stderr_train_losses": stderr_train_losses,
                "val_iters": val_iters,
                "mean_val_losses": mean_val_losses,
                "stderr_val_losses": stderr_val_losses,
                "mean_val_eers": mean_val_eers,
                "stderr_val_eers": stderr_val_eers,
            }

        # Store run_info per run
        results_info[folder] = run_info
    else:
        print(f"Data files not found for run {folder}.")

# Now, plot the data
# Plot 1: Training Loss Across Runs for each dataset
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        run_data = results_info.get(run, {})
        iters = np.array(run_data["train_iters"])
        mean_losses = np.array(run_data["mean_train_losses"])
        stderr_losses = np.array(run_data["stderr_train_losses"])
        label = labels.get(run, run)
        color = colors[i]
        plt.plot(iters, mean_losses, label=label, color=color)
        plt.fill_between(
            iters,
            mean_losses - stderr_losses,
            mean_losses + stderr_losses,
            color=color,
            alpha=0.2,
        )

    plt.title(f"Training Loss Across Runs for {dataset}")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"train_loss_{dataset}_across_runs.png")
    plt.close()
    print(
        f"Training loss plot for {dataset} saved as 'train_loss_{dataset}_across_runs.png'."
    )

# Plot 2: Validation Loss Across Runs for each dataset
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        run_data = results_info.get(run, {})
        iters = np.array(run_data["val_iters"])
        mean_losses = np.array(run_data["mean_val_losses"])
        stderr_losses = np.array(run_data["stderr_val_losses"])
        label = labels.get(run, run)
        color = colors[i]
        plt.plot(iters, mean_losses, label=label, color=color)
        plt.fill_between(
            iters,
            mean_losses - stderr_losses,
            mean_losses + stderr_losses,
            color=color,
            alpha=0.2,
        )

    plt.title(f"Validation Loss Across Runs for {dataset}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"val_loss_{dataset}_across_runs.png")
    plt.close()
    print(
        f"Validation loss plot for {dataset} saved as 'val_loss_{dataset}_across_runs.png'."
    )

# Plot 3: Validation EERs Across Runs for each dataset
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        run_data = results_info.get(run, {})
        iters = np.array(run_data["val_iters"])
        mean_eers = np.array(run_data["mean_val_eers"])
        stderr_eers = np.array(run_data["stderr_val_eers"])
        label = labels.get(run, run)
        color = colors[i]
        plt.plot(iters, mean_eers, label=label, color=color)
        # plt.fill_between(iters, mean_eers - stderr_eers, mean_losses + stderr_eers, color=color, alpha=0.2)

    plt.title(f"Validation EER Across Runs for {dataset}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation EER")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"val_eer_{dataset}_across_runs.png")
    plt.close()
    print(
        f"Validation eer plot for {dataset} saved as 'val_eer_{dataset}_across_runs.png'."
    )

# Plot 4: Test Accuracy Across Runs
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    run_names = []
    accuracies = []
    for i, run in enumerate(runs):
        final_info = final_results.get(run, {})
        dataset_info = final_info.get(dataset, {})
        means = dataset_info.get("means", {})
        test_accuracy = means.get("test_acc_mean", None)
        if test_accuracy is not None:
            run_names.append(labels.get(run, run))
            accuracies.append(test_accuracy)

    if run_names and accuracies:
        plt.bar(
            run_names,
            accuracies,
            color=[
                colors[runs.index(run)]
                for run in runs
                if labels.get(run, run) in run_names
            ],
        )
        plt.title(f"Test Accuracy Across Runs for {dataset}")
        plt.xlabel("Run")
        plt.ylabel("Test Accuracy (%)")
        plt.ylim(0, 100)
        for i, v in enumerate(accuracies):
            plt.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(f"test_accuracy_{dataset}_across_runs.png")
        plt.close()
        print(
            f"Test accuracy plot for {dataset} saved as 'test_accuracy_{dataset}_across_runs.png'."
        )
    else:
        print(f"No test accuracy data available for dataset {dataset}.")

# Plot 5: Test EER Across Runs
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    run_names = []
    accuracies = []
    for i, run in enumerate(runs):
        final_info = final_results.get(run, {})
        dataset_info = final_info.get(dataset, {})
        means = dataset_info.get("means", {})
        test_accuracy = means.get("test_eer_mean", None)
        if test_accuracy is not None:
            run_names.append(labels.get(run, run))
            accuracies.append(test_accuracy)

    if run_names and accuracies:
        plt.bar(
            run_names,
            accuracies,
            color=[
                colors[runs.index(run)]
                for run in runs
                if labels.get(run, run) in run_names
            ],
        )
        plt.title(f"Test EER Across Runs for {dataset}")
        plt.xlabel("Run")
        plt.ylabel("Test EER (%)")
        plt.ylim(0, 100)
        for i, v in enumerate(accuracies):
            plt.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(f"test_eer_{dataset}_across_runs.png")
        plt.close()
        print(
            f"Test eer plot for {dataset} saved as 'test_eer_{dataset}_across_runs.png'."
        )
    else:
        print(f"No test accuracy data available for dataset {dataset}.")
