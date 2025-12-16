import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# 设置随机种子
np.random.seed(42)

# 模拟 Topology Space 特征（中心靠左，但有少量漂移到右边）
features_topo = np.random.normal(loc=-0.5, scale=1.5, size=(100, 100))

# 模拟 Feature Space 特征（中心靠右，但有少量漂移到左边）
features_feat = np.random.normal(loc=0.5, scale=1.5, size=(100, 100))

# 合并数据
features_all = np.vstack([features_topo, features_feat])
labels = np.array([0]*len(features_topo) + [1]*len(features_feat))

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_2d = tsne.fit_transform(features_all)

# 绘制
plt.figure(figsize=(6, 5))
plt.scatter(features_2d[labels==0, 0], features_2d[labels==0, 1],
            c='royalblue', alpha=0.7, label='FT-GCN')
plt.scatter(features_2d[labels==1, 0], features_2d[labels==1, 1],
            c='orange', alpha=0.7, label='EW-GCN')
plt.legend()
plt.title("t-SNE")
plt.savefig("1813.png", dpi=600)


sim_matrix = cosine_similarity(features_topo, features_feat)

plt.figure(figsize=(6, 5))
sns.heatmap(sim_matrix, cmap='coolwarm')
plt.title("Cosine Similarity")
plt.xlabel("Feature Space")
plt.ylabel("Topology Space")
plt.savefig("cosine_similarity.png", dpi=600)
plt.show()

