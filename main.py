import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

class Baum():
    def __init__(self, max_tiefe=5, mode="rand"):
        self.max_tiefe = max_tiefe
        self.tree = None
        self.best_split = self.best_split_rand if "rand" in mode else self.best_split_all

    def build(self, X, y, tiefe=0):
        if tiefe>=self.max_tiefe or len(np.unique(y))<=1:
            #print(tiefe)
            return np.argmax(np.bincount(y))

        split = self.best_split(X, y)
        if split is None:
            #print(tiefe)
            return np.argmax(np.bincount(y))

        feature, threshold = split
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build(X[left_idx], y[left_idx], tiefe + 1),
            "right": self.build(X[right_idx], y[right_idx], tiefe + 1),
        }

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    def best_split_all(self, X, y):
        best_gini = float("inf")
        best_split = None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx

                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                gini_left = self.gini(y[left_idx])
                gini_right = self.gini(y[right_idx])
                gini_split = (left_idx.sum() * gini_left + right_idx.sum() * gini_right) / n_samples

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_split = (feature, threshold)

        return best_split

    def best_split_rand(self, X, y):
        best_gini = float("inf")
        best_split = None
        n_samples, n_features = X.shape

        feature = np.random.randint(0,4)
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_idx = X[:, feature] <= threshold
            right_idx = ~left_idx

            if left_idx.sum() == 0 or right_idx.sum() == 0:
                continue

            gini_left = self.gini(y[left_idx])
            gini_right = self.gini(y[right_idx])
            gini_split = (left_idx.sum() * gini_left + right_idx.sum() * gini_right) / n_samples

            if gini_split < best_gini:
                best_gini = gini_split
                best_split = (feature, threshold)

        return best_split

    def fit(self, X, y):
        self.tree = self.build(X, y)

    def predict_sample(self, x, node):
        if not isinstance(node, dict):
            return node

        if x[node["feature"]] <= node["threshold"]:
            return self.predict_sample(x, node["left"])
        else:
            return self.predict_sample(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])

    def draw(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-2,2)
        ax.set_ylim(0,1)
        ax.axis("off")

        def zeichne(node, x, y, dx):
            if isinstance(node, dict):
                #ax.scatter(x, y, s=300, color="lightblue", edgecolor="black", zorder=3)
                ax.text(x, y, f"$X_{node['feature']}$>{node['threshold']:.1f}", fontsize=8, ha="center", va="center",bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))

                x_links = x - dx
                y_links = y - 0.2
                ax.plot([x-0.05, x_links], [y-0.025, y_links+0.025], color="black", lw=1)
                if isinstance(node["left"], dict):
                    zeichne(node["left"], x_links, y_links, dx / 2)
                else:
                    ax.text(x_links, y_links, str(node["left"]), fontsize=10, ha="center", va="center", 
                            bbox=dict(boxstyle="round",
                                    ec=(0.5, 0.5, 1.),
                                    fc=(0.8, 0.8, 1.),
                   ))

                x_rechts = x + dx
                y_rechts = y - 0.2
                ax.plot([x+0.05, x_rechts], [y-0.025, y_rechts+0.025], color="black", lw=1)
                if isinstance(node["right"], dict):
                    zeichne(node["right"], x_rechts, y_rechts, dx / 2)
                else:
                    ax.text(x_rechts, y_rechts, str(node["right"]), fontsize=10, ha="center", va="center", 
                            bbox=dict(boxstyle="round",
                                    ec=(0.5, 0.5, 1.),
                                    fc=(0.8, 0.8, 1.),
                   ))

        zeichne(self.tree, 0,0.9,1)
        st.pyplot(fig)


data, label = np.load("data.npy"), np.load("label.npy")
idc = np.random.permutation(len(data))[:int(0.8*len(data))]
data, label = data[idc], label[idc]

baum = Baum(max_tiefe=4, mode="rand")
baum.fit(data, label)

st.title(":seedling: Eigener Entscheidungsbaum :deciduous_tree:")
baum.draw()