import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, mean_squared_error
)
from scipy.stats import ttest_rel
import warnings

warnings.filterwarnings("ignore")


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=0.1, momentum=0.9, weight_decay=0.001,
                 task="classification"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.task = task

        # Xavier init
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))

        # Momentum terms
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.a1 = np.dot(X, self.W1) + self.b1
        self.h1 = self.activation(self.a1)
        self.a2 = np.dot(self.h1, self.W2) + self.b2
        if self.task == "classification":
            self.output = self.activation(self.a2)
        else:
            self.output = self.a2
        return self.output

    def forward_pass_internal(self, X):
        a1 = np.dot(X, self.W1) + self.b1
        h1 = self.activation(a1)
        a2 = np.dot(h1, self.W2) + self.b2
        if self.task == "classification":
            output = self.activation(a2)
        else:
            output = a2
        return output, h1

    def backward(self, X, y, output):
        if self.task == "classification":
            error = output - y
            d_output = error
            d_h1 = np.dot(d_output, self.W2.T) * self.activation_derivative(self.h1)
        else:
            error = output - y
            d_output = 2 * error / X.shape[0]
            d_h1 = np.dot(d_output, self.W2.T) * self.activation_derivative(self.h1)

        dW2 = np.dot(self.h1.T, d_output) + self.weight_decay * self.W2
        db2 = np.sum(d_output, axis=0, keepdims=True)
        dW1 = np.dot(X.T, d_h1) + self.weight_decay * self.W1
        db1 = np.sum(d_h1, axis=0, keepdims=True)

        self.vW2 = self.momentum * self.vW2 - self.learning_rate * dW2
        self.vb2 = self.momentum * self.vb2 - self.learning_rate * db2
        self.vW1 = self.momentum * self.vW1 - self.learning_rate * dW1
        self.vb1 = self.momentum * self.vb1 - self.learning_rate * db1

        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1

    def train(self, X_train, y_train, epochs=1000):
        for _ in range(epochs):
            output = self.forward(X_train)
            self.backward(X_train, y_train, output)


def prepare_datasets():
    datasets = {"classification": {}, "regression": {}}

    # Classification
    X_clf1, y_clf1 = make_classification(
        n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42
    )
    X_clf2, y_clf2 = make_classification(
        n_samples=1000, n_features=5, n_informative=3, n_redundant=1, n_classes=2, random_state=42
    )
    X_clf3, y_clf3 = make_classification(
        n_samples=2000, n_features=10, n_informative=5, n_redundant=2, n_classes=3, random_state=42
    )

    # Regression
    X_reg1, y_reg1 = make_regression(n_samples=500, n_features=1, noise=20, random_state=42)
    X_reg2, y_reg2 = make_regression(n_samples=1000, n_features=5, noise=50, random_state=42)
    X_reg3, y_reg3 = make_regression(n_samples=2000, n_features=10, noise=100, random_state=42)

    scaler = StandardScaler()
    datasets["classification"] = {
        "clf1": (scaler.fit_transform(X_clf1), y_clf1),
        "clf2": (scaler.fit_transform(X_clf2), y_clf2),
        "clf3": (scaler.fit_transform(X_clf3), y_clf3),
    }
    datasets["regression"] = {
        "reg1": (scaler.fit_transform(X_reg1), y_reg1),
        "reg2": (scaler.fit_transform(X_reg2), y_reg2),
        "reg3": (scaler.fit_transform(X_reg3), y_reg3),
    }
    return datasets


def alus_sampling(nn, X_pool, n_query=10):
    probas = nn.forward(X_pool)
    if nn.output_size == 1:  # binary
        uncertainty = 1 - np.abs(probas - 0.5)
    else:
        probas_sorted = np.sort(probas, axis=1)
        margin = probas_sorted[:, -1] - probas_sorted[:, -2]
        entropy = -np.sum(probas * np.log(probas + 1e-9), axis=1)
        uncertainty = (1 - margin) * (1 + entropy)
    return np.argsort(uncertainty)[-n_query:]


def sasla_sampling(nn, X_pool, n_query=10):
    output, h1 = nn.forward_pass_internal(X_pool)
    d_output_d_a2 = nn.activation_derivative(output)
    d_h1_d_a1 = nn.activation_derivative(h1)
    sensitivity = np.dot(np.dot(d_output_d_a2, nn.W2.T) * d_h1_d_a1, nn.W1.T)
    # Use the infinity norm 
    sensitivity_norm = np.max(np.abs(sensitivity), axis=1)
    return np.argsort(sensitivity_norm)[-n_query:]


def evaluate_classification(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba)
    return metrics


def run_experiments(datasets, n_trials=10, initial_pool_size=0.1, query_size=10, n_queries=10):
    results = {"classification": {}, "regression": {}}

    for task_type, task_datasets in datasets.items():
        for dataset_name, (X, y) in task_datasets.items():
            results[task_type][dataset_name] = {"passive": [], "alus": [], "sasla": []}

            for trial in range(n_trials):
                X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=trial)
                if task_type == "classification":
                    n_classes = len(np.unique(y_train_full))
                    y_train_full_encoded = np.eye(n_classes)[y_train_full.astype(int)]
                else:
                    y_train_full_encoded = y_train_full.reshape(-1, 1)

                input_size = X_train_full.shape[1]
                hidden_size = 50
                output_size = len(np.unique(y)) if task_type == "classification" else 1

                # Passive
                nn_passive = NeuralNetwork(input_size, hidden_size, output_size, task=task_type)
                nn_passive.train(X_train_full, y_train_full_encoded, epochs=1000)
                pred = nn_passive.forward(X_test)
                if task_type == "classification":
                    y_pred = np.argmax(pred, axis=1)
                    metrics = evaluate_classification(y_test, y_pred, pred)
                else:
                    metrics = {"mse": mean_squared_error(y_test, pred)}
                metrics["patterns"] = X_train_full.shape[0]
                results[task_type][dataset_name]["passive"].append(metrics)

                # Split pools
                X_pool, X_labeled, y_pool, y_labeled = train_test_split(
                    X_train_full, y_train_full, test_size=initial_pool_size, random_state=trial
                )
                if task_type == "classification":
                    y_labeled_encoded = np.eye(n_classes)[y_labeled.astype(int)]
                else:
                    y_labeled_encoded = y_labeled.reshape(-1, 1)

                # ALUS
                nn_alus = NeuralNetwork(input_size, hidden_size, output_size, task=task_type)
                X_alus, y_alus = X_labeled.copy(), y_labeled_encoded.copy()
                X_alus_pool, y_alus_pool = X_pool.copy(), y_pool.copy()
                for _ in range(n_queries):
                    nn_alus.train(X_alus, y_alus, epochs=50)
                    query_indices = alus_sampling(nn_alus, X_alus_pool, query_size)
                    X_alus = np.vstack([X_alus, X_alus_pool[query_indices].reshape(len(query_indices), -1)])
                    if task_type == "classification":
                        y_alus = np.vstack([y_alus, np.eye(n_classes)[y_alus_pool[query_indices].astype(int)]])
                    else:
                        y_alus = np.vstack([y_alus, y_alus_pool[query_indices].reshape(len(query_indices), -1)])
                    X_alus_pool = np.delete(X_alus_pool, query_indices, axis=0)
                    y_alus_pool = np.delete(y_alus_pool, query_indices, axis=0)
                pred = nn_alus.forward(X_test)
                if task_type == "classification":
                    y_pred = np.argmax(pred, axis=1)
                    metrics = evaluate_classification(y_test, y_pred, pred)
                else:
                    metrics = {"mse": mean_squared_error(y_test, pred)}
                metrics["patterns"] = X_alus.shape[0]
                results[task_type][dataset_name]["alus"].append(metrics)

                # SASLA
                nn_sasla = NeuralNetwork(input_size, hidden_size, output_size, task=task_type)
                X_sasla, y_sasla = X_labeled.copy(), y_labeled_encoded.copy()
                X_sasla_pool, y_sasla_pool = X_pool.copy(), y_pool.copy()
                for _ in range(n_queries):
                    nn_sasla.train(X_sasla, y_sasla, epochs=50)
                    query_indices = sasla_sampling(nn_sasla, X_sasla_pool, query_size)
                    X_sasla = np.vstack([X_sasla, X_sasla_pool[query_indices].reshape(len(query_indices), -1)])
                    if task_type == "classification":
                        y_sasla = np.vstack([y_sasla, np.eye(n_classes)[y_sasla_pool[query_indices].astype(int)]])
                    else:
                        y_sasla = np.vstack([y_sasla, y_sasla_pool[query_indices].reshape(len(query_indices), -1)])
                    X_sasla_pool = np.delete(X_sasla_pool, query_indices, axis=0)
                    y_sasla_pool = np.delete(y_sasla_pool, query_indices, axis=0)
                pred = nn_sasla.forward(X_test)
                if task_type == "classification":
                    y_pred = np.argmax(pred, axis=1)
                    metrics = evaluate_classification(y_test, y_pred, pred)
                else:
                    metrics = {"mse": mean_squared_error(y_test, pred)}
                metrics["patterns"] = X_sasla.shape[0]
                results[task_type][dataset_name]["sasla"].append(metrics)

    return results


if __name__ == "__main__":
    datasets = prepare_datasets()
    results = run_experiments(datasets, n_trials=10)

    print("\n=== SUMMARY STATISTICS ===")
    for task_type, task_datasets in results.items():
        print(f"\n{task_type.upper()} TASKS:")
        for dataset_name, methods in task_datasets.items():
            print(f"\nDataset: {dataset_name}")
            passive_metrics = methods["passive"]
            for method, metrics_list in methods.items():
                all_keys = metrics_list[0].keys()
                summary = {}
                for key in all_keys:
                    values = [m[key] for m in metrics_list if key in m]
                    summary[key] = (np.mean(values), np.std(values))
                    # Paired t-test vs Passive
                    if method != "passive":
                        passive_vals = [m[key] for m in passive_metrics if key in m]
                        if len(passive_vals) == len(values):
                            _, pval = ttest_rel(passive_vals, values)
                            summary[key] += (pval,)
                print(f"  {method}:")
                for key, vals in summary.items():
                    mean, std = vals[0], vals[1]
                    if key == "patterns":
                        mean = np.mean(values)
                        print(f"    {key}: {mean:.0f}")
                    elif len(vals) == 3:
                        pval = vals[2]
                        print(f"    {key}: {mean:.4f} ± {std:.4f}, p={pval:.4f}")
                    else:
                        print(f"    {key}: {mean:.4f} ± {std:.4f}")
