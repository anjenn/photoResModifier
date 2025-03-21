import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def apply_sepia(image):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    return np.clip(image @ sepia_filter.T, 0, 255).astype(np.uint8)

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distance = []
        for j in range(K):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)
    
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    
    centroids = np.zeros((K, n))

    for k in range(K):   
        points = X[idx == k] # to get a list of all data points in X assigned to centroid k  
        centroids[k] = np.mean(points, axis = 0) # to compute the mean of the points assigned
    
    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=True):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    for i in range(max_iters):

        print("K-Means iteration %d/%d" % (i, max_iters-1))
        idx = find_closest_centroids(X, centroids)

        centroids = compute_centroids(X, idx, K)
        if plot_progress and (i % 2 == 0 or i == max_iters - 1):  
            X_recovered = centroids[idx, :]
            X_recovered = np.reshape(X_recovered, (256, 256, 3))

            plt.imshow(X_recovered.astype(np.uint8))
            plt.title(f"Iteration {i+1}")
            plt.axis('off')
            plt.show()

    return centroids, idx

def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    
    return centroids

K = 16
max_iters = 10

original_img = Image.open('./ctn_01.jpg')

resized_img = original_img.resize((256, 256), Image.LANCZOS)
resized_array = np.array(resized_img)

########################################################################
# Experiementing with compression across different color filters
sepia_image = apply_sepia(resized_array)
X_sepia = np.reshape(sepia_image, (sepia_image.shape[0] * sepia_image.shape[1], 3))
X_img = X_sepia
########################################################################

X_img = np.reshape(resized_array, (resized_array.shape[0] * resized_array.shape[1], 3))

# Using the function you have implemented above. 
initial_centroids = kMeans_init_centroids(X_img, K)

# Run K-Means - this can take a couple of minutes depending on K and max_iters
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

# Find the closest centroid of each pixel
idx = find_closest_centroids(X_img, centroids)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx.astype(int), :] 


X_recovered = np.clip(centroids[idx.astype(int), :], 0, 255)  # Ensure values are within valid range

# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, resized_array.shape) 

X_recovered = X_recovered.astype(np.uint8)  # Convert to integer format for display


# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,8))
plt.axis('off')

ax[0].imshow(resized_array)
ax[0].set_title('Original')
ax[0].axis('off')

# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].axis('off')
plt.show()

# Calculating distortions
distortions = []
for i in range(max_iters):
    idx = find_closest_centroids(X_img, centroids)
    centroids = compute_centroids(X_img, idx, K)
    distortion = np.sum((X_img - centroids[idx])**2)
    distortions.append(distortion)

plt.plot(range(max_iters), distortions, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Distortion')
plt.title('K-Means Convergence')
plt.show()