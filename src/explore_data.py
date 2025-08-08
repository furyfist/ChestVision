# In src/explore_data.py
from datasets import load_dataset

# This line downloads (if not already cached) and loads the data
ds = load_dataset("dorsar/lung-cancer")

# Now, let's inspect it
print(ds)

# get the first training example
example = ds['train'][0]
image = example['image']

# print the image details
print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")

# print the features, which includes the label names
print(ds['train'].features)
