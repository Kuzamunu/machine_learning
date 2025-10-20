import os
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
if not os.path.exists('results'):
    os.makedirs('results')

def generate_datasets():
    n_samples = 5000
    n_features = 20
    # Linear dataset
    X_linear, y_linear = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=5,
        n_redundant=5, n_clusters_per_class=1, random_state=42
    )
    # Moons dataset
    X_moons_base, y_moons = make_moons(
        n_samples=n_samples, noise=0.3, random_state=42
    )
    X_moons_extra = np.random.normal(0, 1, (n_samples, n_features - 2))
    X_moons = np.hstack([X_moons_base, X_moons_extra])
    # Circles dataset
    X_circles_base, y_circles = make_circles(
        n_samples=n_samples, noise=0.4, factor=0.5, random_state=42
    )
    X_circles_extra = np.random.normal(0, 1, (n_samples, n_features - 2))
    X_circles = np.hstack([X_circles_base, X_circles_extra])
    return {
        "Linear (make_classification)": (X_linear, y_linear),
        "Nonlinear (make_moons - 20D)": (X_moons, y_moons),
        "Nonlinear (make_circles - 20D)": (X_circles, y_circles)
    }
datasets = generate_datasets()

tree_depths = [1, 2, 3, 4, 5, 7, 10, 15, 17, None]
max_features_list = [1, 2, 3, 5, 7, 'sqrt', 'log2', None]
n_trees_list = [10, 50, 100, 200, 300]
n_runs = 10
all_results = {}

for name, (X, y) in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Data shape: {X.shape}")
    print(f"{'='*60}")

    dataset_results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # EXPERIMENT 1
    print("\n[1] Varying tree depth (fixed n_estimators=100, max_features='sqrt')")
    depth_scores_mean = []
    depth_scores_std = []

    for depth in tree_depths:
        run_scores = []
        for run in range(n_runs):
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=depth,
                max_features='sqrt', random_state=42+run, n_jobs=-1
            )
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            run_scores.append(acc)

        depth_scores_mean.append(np.mean(run_scores))
        depth_scores_std.append(np.std(run_scores))

    plt.figure(figsize=(8, 5))
    plt.errorbar([str(d) for d in tree_depths], depth_scores_mean, yerr=depth_scores_std, 
                 marker='o', capsize=5, linewidth=2)
    plt.title(f"Effect of Tree Depth - {name}", fontsize=14, fontweight='bold')
    plt.xlabel("Tree Depth", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/exp1_tree_depth_{name.replace(" ", "_").replace("(", "").replace(")", "")}.png', dpi=300, bbox_inches='tight')
    plt.close()

    best_depth_idx = np.argmax(depth_scores_mean)
    best_depth = tree_depths[best_depth_idx]
    best_depth_acc = depth_scores_mean[best_depth_idx]
    print(f"Best depth for {name}: {best_depth} (Acc={best_depth_acc:.3f} ± {depth_scores_std[best_depth_idx]:.3f})")

    dataset_results['exp1'] = {
        'depths': [str(d) for d in tree_depths],
        'mean_scores': depth_scores_mean,
        'std_scores': depth_scores_std,
        'best_depth': str(best_depth),
        'best_depth_acc': best_depth_acc
    }

    # EXPERIMENT 2
    print("\n[2] Relationship between tree depth and number of trees")
    depth_tree_matrix = np.zeros((len(tree_depths), len(n_trees_list)))

    for i, depth in enumerate(tree_depths):
        for j, n_trees in enumerate(n_trees_list):
            run_scores = []
            for run in range(n_runs):
                clf = RandomForestClassifier(
                    n_estimators=n_trees, max_depth=depth,
                    max_features='sqrt', random_state=42+run, n_jobs=-1
                )
                clf.fit(X_train, y_train)
                acc = accuracy_score(y_test, clf.predict(X_test))
                run_scores.append(acc)
            depth_tree_matrix[i, j] = np.mean(run_scores)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(depth_tree_matrix, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(im, label='Accuracy')
    plt.xticks(np.arange(len(n_trees_list)), n_trees_list)
    plt.yticks(np.arange(len(tree_depths)), [str(d) for d in tree_depths])
    plt.xlabel("Number of Trees (n_estimators)", fontsize=12)
    plt.ylabel("Tree Depth", fontsize=12)
    plt.title(f"Depth vs Number of Trees - {name}", fontsize=14, fontweight='bold')

    for i in range(len(tree_depths)):
        for j in range(len(n_trees_list)):
            plt.text(j, i, f'{depth_tree_matrix[i, j]:.3f}', 
                    ha="center", va="center", color="white" if depth_tree_matrix[i, j] < np.median(depth_tree_matrix) else "black")

    plt.tight_layout()
    plt.savefig(f'results/exp2_depth_vs_trees_{name.replace(" ", "_").replace("(", "").replace(")", "")}.png', dpi=300, bbox_inches='tight')
    plt.close()

    dataset_results['exp2'] = {
        'matrix': depth_tree_matrix.tolist(),
        'tree_depths': [str(d) for d in tree_depths],
        'n_trees_list': n_trees_list
    }

    # EXPERIMENT 3
    print(f"\n[3] Varying max_features (fixed best_depth={best_depth}, n_estimators=100)")
    feat_scores_mean = []
    feat_scores_std = []

    for mf in max_features_list:
        run_scores = []
        for run in range(n_runs):
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=best_depth,
                max_features=mf, random_state=42+run, n_jobs=-1
            )
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            run_scores.append(acc)

        feat_scores_mean.append(np.mean(run_scores))
        feat_scores_std.append(np.std(run_scores))

    plt.figure(figsize=(8, 5))
    plt.errorbar([str(mf) for mf in max_features_list], feat_scores_mean, yerr=feat_scores_std,
                 marker='o', capsize=5, linewidth=2, color='orange')
    plt.title(f"Effect of Max Features - {name}\n(Fixed depth={best_depth})", fontsize=14, fontweight='bold')
    plt.xlabel("max_features", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/exp3_max_features_{name.replace(" ", "_").replace("(", "").replace(")", "")}.png', dpi=300, bbox_inches='tight')
    plt.close()

    best_feat_idx = np.argmax(feat_scores_mean)
    best_feat = max_features_list[best_feat_idx]
    best_feat_acc = feat_scores_mean[best_feat_idx]
    print(f"Best max_features for {name}: {best_feat} (Acc={best_feat_acc:.3f} ± {feat_scores_std[best_feat_idx]:.3f})")

    dataset_results['exp3'] = {
        'features': [str(mf) for mf in max_features_list],
        'mean_scores': feat_scores_mean,
        'std_scores': feat_scores_std,
        'best_feature': str(best_feat),
        'best_feature_acc': best_feat_acc
    }

    # EXPERIMENT 4
    print("\n[4] A mix of everything, a little bit of this and that")
    grid_scores = np.zeros((len(tree_depths), len(max_features_list)))

    for i, depth in enumerate(tree_depths):
        for j, mf in enumerate(max_features_list):
            run_scores = []
            for run in range(n_runs):
                clf = RandomForestClassifier(
                    n_estimators=100, max_depth=depth,
                    max_features=mf, random_state=42+run, n_jobs=-1
                )
                clf.fit(X_train, y_train)
                acc = accuracy_score(y_test, clf.predict(X_test))
                run_scores.append(acc)
            grid_scores[i, j] = np.mean(run_scores)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(grid_scores, cmap='plasma', aspect='auto', origin='lower')
    plt.colorbar(im, label='Accuracy')
    plt.xticks(np.arange(len(max_features_list)), [str(mf) for mf in max_features_list])
    plt.yticks(np.arange(len(tree_depths)), [str(d) for d in tree_depths])
    plt.xlabel('max_features', fontsize=12)
    plt.ylabel('max_depth', fontsize=12)
    plt.title(f"Combined Effect - {name}", fontsize=14, fontweight='bold')

    best_idx = np.unravel_index(grid_scores.argmax(), grid_scores.shape)
    plt.plot(best_idx[1], best_idx[0], 'rx', markersize=15, markeredgewidth=3)

    plt.tight_layout()
    plt.savefig(f'results/exp4_combined_effect_{name.replace(" ", "_").replace("(", "").replace(")", "")}.png', dpi=300, bbox_inches='tight')
    plt.close()

    best_idx = np.unravel_index(grid_scores.argmax(), grid_scores.shape)
    best_combo_depth = tree_depths[best_idx[0]]
    best_combo_feat = max_features_list[best_idx[1]]
    best_combo_acc = grid_scores.max()

    print(f"Best Combo → depth={best_combo_depth}, max_features={best_combo_feat}, acc={best_combo_acc:.3f}")

    dataset_results['exp4'] = {
        'grid_scores': grid_scores.tolist(),
        'best_combo_depth': str(best_combo_depth),
        'best_combo_feat': str(best_combo_feat),
        'best_combo_acc': best_combo_acc
    }
    
    all_results[name] = dataset_results

with open('results/summary.json', 'w') as f:
    json.dump(all_results, f, indent=4)
print("finished")

