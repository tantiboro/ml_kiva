import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report


def plot_learning_curve(pipeline, X, y, save_path):
    """Generates and saves the scaling analysis plot."""
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        train_sizes, np.mean(train_scores, axis=1), "o-", label="Training Accuracy"
    )
    plt.plot(
        train_sizes, np.mean(test_scores, axis=1), "s-", label="Validation Accuracy"
    )
    plt.title("Learning Curve: Scaling Analysis")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"📊 Learning curve saved to {save_path}")


def print_feature_importance(pipeline):
    """Extracts and prints the most predictive features."""
    preprocessor = pipeline.named_steps["preprocessor"]

    tfidf_names = preprocessor.named_transformers_["text"].get_feature_names_out()
    num_names = ["loan_amount", "lender_term"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out()

    all_names = np.concatenate([tfidf_names, num_names, cat_names])
    importances = pipeline.named_steps["rf"].feature_importances_

    feat_df = pd.DataFrame({"feature": all_names, "importance": importances})
    print("\n🏆 Top 20 Most Predictive Features:")
    print(feat_df.sort_values(by="importance", ascending=False).head(20))
