import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.util import img_as_float

# Load the image
img_path = r"C:\dataset\vindr-pcxr-an-open-large-scale-pediatric-chest-x-ray-dataset-for-interpretation-of-common-thoracic-diseases-1.0.0\val_png\images\0aa39f21f667a341b2b8b0f3db5708a4.png"
img = io.imread(img_path, as_gray=True)

# Display the original image
plt.figure(figsize=(10, 8))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# ### Min-Max Normalization
min_max_normalized_img = (img - img.min()) / (img.max() - img.min())
plt.subplot(2, 3, 2)
plt.imshow(min_max_normalized_img, cmap='gray')
plt.title('Min-Max Normalization')
plt.axis('off')

# ### Z-Score Normalization
z_score_normalized_img = (img - img.mean()) / img.std()
plt.subplot(2, 3, 3)
plt.imshow(z_score_normalized_img, cmap='gray')
plt.title('Z-Score Normalization')
plt.axis('off')

# ### Histogram Equalization
hist_equalized_img = exposure.equalize_hist(img)
plt.subplot(2, 3, 4)
plt.imshow(hist_equalized_img, cmap='gray')
plt.title('Histogram Equalization')
plt.axis('off')

# ### CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe_img = exposure.equalize_adapthist(img, clip_limit=0.03)
plt.subplot(2, 3, 5)
plt.imshow(clahe_img, cmap='gray')
plt.title('CLAHE')
plt.axis('off')

# ### Float Normalization
float_normalized_img = img_as_float(img)
plt.subplot(2, 3, 6)
plt.imshow(float_normalized_img, cmap='gray')
plt.title('Float Normalization')
plt.axis('off')

plt.tight_layout()
plt.show()