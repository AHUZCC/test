from sklearn.datasets import load_digits
digits = load_digits()

x_digits,y_digits = digits.data,digits.target

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

estimator = PCA(n_components= 2)

x_pca = estimator.fit_transform(x_digits)
# 聚类问题经常需要直观的展现数据，降维度的一个直接目的也为此；
#因此我们这里多展现几个图片直观一些。

def plot_pca_scatter():
    colors = ['back','blue','purple','yellow','white','red','lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = x_pca[:,0][y_digits==i]
        py = x_pca[:,1][y_digits==i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
    
plot_pca_scatter()