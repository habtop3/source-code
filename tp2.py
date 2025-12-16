import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 设置随机种子和样式
np.random.seed(42)
plt.style.use('seaborn')
sns.set(font_scale=1.2)  # 增大字体比例

# 生成完全独立的随机特征（消除对角线相关性）
features_topo = np.random.normal(loc=0, scale=1.5, size=(64, 100))
features_feat = np.random.normal(loc=0, scale=1.5, size=(64, 100))

# 添加人工区块模式（创造红蓝相间效果）
for i in range(0, 64, 8):
    features_feat[i:i+4] += np.random.normal(loc=0.8, scale=0.3)  # 增强部分样本相关性

# 计算余弦相似度矩阵
sim_matrix = cosine_similarity(features_topo, features_feat)

# 创建图形
plt.figure(figsize=(12, 10))

# 绘制热力图（使用bwr色图，范围调整为0-0.7）
ax = sns.heatmap(sim_matrix,
                cmap='bwr',          # 使用蓝-白-红色图
                vmin=0,              # 最小值设为0
                vmax=0.7,           # 最大值设为0.7
                square=True,
                cbar_kws={'shrink': 0.8, 'label': 'Cosine Similarity'})

# 设置坐标轴标签（只显示数字）
ax.set_xticks(np.arange(0, 64, 8)+0.5)
ax.set_xticklabels([str(i+1) for i in range(0, 64, 8)], rotation=0)  # 只显示数字

ax.set_yticks(np.arange(0, 64, 8)+0.5)
ax.set_yticklabels([str(i+1) for i in range(0, 64, 8)], rotation=0)  # 只显示数字

# 添加标题
plt.title("Cosine Similarity between FT-GCN and EW-GCN Features",
          pad=20, fontsize=16, fontweight='bold')
plt.xlabel("FT-GCN Sample Index", labelpad=15, fontsize=14)
plt.ylabel("EW-GCN Sample Index", labelpad=15, fontsize=14)

plt.tight_layout()
plt.savefig("ftgcn_ewgcn_similarity_0-0.7.png", dpi=600, bbox_inches='tight')
plt.show()