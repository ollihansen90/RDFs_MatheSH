import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numbers import Number

class Baum():
    def __init__(self, max_tiefe=5, mode="rand", root=None):
        self.max_tiefe = max_tiefe
        self.tree = None
        self.left = None
        self.right = None
        if mode=="rand":
            self.best_split = self.best_split_rand 
        elif mode=="own":
            self.best_split = self.split
        else:
            self.best_split = self.best_split_all
        if root is not None:
            self.root = root
        else:
            self.root = self
    
    def build(self, X, y, tiefe=0):
        if tiefe>=self.max_tiefe or len(np.unique(y))==0:
            return np.argmax(np.bincount(y))

        split = "nochmal"
        split = self.best_split(X, y)
        while split=="nochmal":
            split = self.best_split(X, y)
            
        if split is None:
            return np.argmax(np.bincount(y))

        feature, threshold = split
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        self.feature = feature
        self.threshold = threshold
        
        self.left = Baum(max_tiefe=self.max_tiefe, mode="own", root=self.root)
        self.left.build(X[left_idx], y[left_idx], tiefe + 1)
        self.right = Baum(max_tiefe=self.max_tiefe, mode="own", root=self.root)
        self.right.build(X[right_idx], y[right_idx], tiefe + 1)
        st.session_state.root = self.root.get_state()

    def get_state(self):
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left.get_state() if isinstance(self.left, Baum) else self.left,
            "right": self.right.get_state() if isinstance(self.right, Baum) else self.right,
        }
    
    def set_state(self, state):
        self.feature = state["feature"]
        self.threshold = state["threshold"]
        if isinstance(state["left"], Number):
            self.left = state["left"]
        else:
            self.left = Baum(max_tiefe=self.max_tiefe, mode="own", root=self.root)
            self.left.set_state(state["left"])
        if isinstance(state["right"], Number):
            self.right = state["right"]
        else:
            self.right = Baum(max_tiefe=self.max_tiefe, mode="own", root=self.root)
            self.right.set_state(state["right"])

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)
    
    def split(self, X, y):
        if "feature" not in st.session_state:
            st.session_state.feature = np.random.randint(0,len(X[0]))
        if "splitvalue" not in st.session_state:
            st.session_state.splitvalue = np.mean(X[:,st.session_state.feature])  # Standardwert des Sliders
        fig, ax = plt.subplots()
        ax.scatter(X[y==0,st.session_state.feature], X[y==0,(st.session_state.feature+1)%len(X[0])], c="tab:blue", alpha=0.8)
        ax.scatter(X[y==1,st.session_state.feature], X[y==1,(st.session_state.feature+1)%len(X[0])], c="tab:orange", alpha=0.8)
        ax.axvline(x=st.session_state.splitvalue, color="red", linestyle="--")
        ax.set_xlabel(f"X{st.session_state.feature}")
        ax.set_ylabel(f"X{(st.session_state.feature+1)%len(X[0])}")
        ax.grid()
        st.pyplot(fig, clear_figure=True)
        # slider zum Eintragen des Splits:
        st.session_state.splitvalue = st.slider(f"X{st.session_state.feature}", min_value=min(X[:,0]), max_value=max(X[:,0]), value=np.mean(X[:,st.session_state.feature]))
        st.write(f"Links: {self.gini(y[X[:,st.session_state.feature]<=st.session_state.splitvalue]):.4f}")
        st.write(f"Rechts: {self.gini(y[X[:,st.session_state.feature]>st.session_state.splitvalue]):.4f}")
        if st.button("Okay", type="primary"):
            feature = st.session_state.feature
            splitvalue = st.session_state.splitvalue
            st.session_state.feature = np.random.randint(0,len(X[0]))
            return feature, splitvalue
        else:
            return "nochmal"

        

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
        if not isinstance(node, Baum):
            return node

        if x[node["feature"]] <= node["threshold"]:
            return self.predict_sample(x, node["left"])
        else:
            return self.predict_sample(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])
    
baum = Baum(max_tiefe=3, mode="own")
st.title("Eigener Entscheidungsbaum")

if not hasattr(st.session_state, "root"):
    st.session_state.data = np.load("data.npy")
    st.session_state.label = np.load("label.npy")
    st.session_state.data_test = np.load("data_test.npy")
    st.session_state.label_test = np.load("label_test.npy")

    st.session_state.idc = np.random.permutation(len(st.session_state.data))[:int(0.8*len(st.session_state.data))]
    st.session_state.data = st.session_state.data[st.session_state.idc]
    st.session_state.label = st.session_state.label[st.session_state.idc]

else:
    baum.set_state(st.session_state.root)

baum.fit(st.session_state.data, st.session_state.label)

