from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
import pandas as pd

df = pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=["label"])], axis=1)
print(df.head())

# Display the first 10 images
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components= 81)
X_pca = pca.fit_transform(X)    

# Display the first 10 images after PCA
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_pca[i].reshape(9, 9), cmap='gray')
    plt.axis('off')
plt.show()  

X_reconstructed = pca.inverse_transform(X_pca)
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()