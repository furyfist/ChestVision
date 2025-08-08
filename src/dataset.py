import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LungCancerDataset(Dataset):
    def __init__(self, hg_dataset_split):

        """
        Args:
            hg_dataset_split: A split of a Hugging Face dataset (e.g., raw_ds['train']).
        """

        # Store the specific dataset split (eg. train, validation or test)
        self.dataset = hg_dataset_split

        # Define the pipeline of image transformations
        self.transform = transforms.Compose([
            # Resize the image to a fixed square size (224 x 224)
            transforms.Resize((224,224)),

            # Convert to pytorch tensors
            transforms.ToTensor(),

            # Normalize the tensors values, values are from models pre-trained on ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        """
        Retrieves one sample from the dataset at a given index 'idx'

        This involves:
        1. Getting the raw image and label
        2. Converting the image from RGBA to RGB
        3. Applying the defined transformations
        4. Returning the transformed image and its label as tensors
        """

        # get the raw image
        item = self.dataset[idx]

        # extract image and label
        image = item['image']
        label = item['label']

        # The images are in RGBA format; convert them to RGB
        # The 'A' (Alpha/transparency) channel is not needed for most image models
        rgb_image = image.convert('RGB')

        # apply the transformation pipeline (resize, to-tensor, normalize)
        transformed_image = self.transform(rgb_image)

        # Return the transformed image and its corresponding label
        # The label is converted to a tensor with dtype=long, as required by PyTorch's loss function
        return transformed_image, torch.tensor(label, dtype=torch.long)
        