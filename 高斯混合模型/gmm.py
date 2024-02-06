from sklearn.mixture import GaussianMixture
import numpy as np

# 生成模拟数据，这里假设我们有一个二元高斯混合模型
np.random.seed(0)
X = np.r_[2 * np.random.randn(100, 2), np.random.randn(100, 2)]

# 初始化高斯混合模型
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)

# 输出模型参数
print("GMM parameters:")
print("Means:", gmm.means_)
print("Covariance matrices:", gmm.covariances_)
print("Concentrations:", gmm.weights_)