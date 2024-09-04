from pathlib import Path
from copy import deepcopy
import json

from tqdm.autonotebook import tqdm, trange

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np
from numpy.lib.format import open_memmap

import pandas as pd

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision.transforms.v2 import AutoAugment, Compose
from torchvision.io import read_image, ImageReadMode
from torchvision.datasets import OxfordIIITPet
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.string_labels = labels
        self.labels = torch.tensor([[1. if label_string == 'cat' else 0.] for label_string in self.string_labels])
        self.transforms = transforms
    
    def set_transforms(self, transforms):
        self.transforms = transforms
        
    def __getitem__(self, index):
        image_path = self.images[index]
        image = read_image(image_path, mode=ImageReadMode.RGB)
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[index]
        return image, label
    
    def __len__(self):
        return len(self.images)
    

class PreAugmentedDataset(Dataset):
    def __init__(self, dataset_dir, files_subset, labels_subset):
        self.dataset_dir = dataset_dir
        self.epoch = 0
        
        # We load the array as a memory mapped array, this makes it lazily read the data as we're 
        # requesting it and allows the operating system to handle flushing pages from the dataset we haven't used
        self.data =  np.load(dataset_dir / 'data.npy', mmap_mode='r')
        self.max_epochs, self.samples, self.n_channels, self.height, self.width = self.data.shape
        self.labels_subset = labels_subset
        self.labels =  np.load(dataset_dir / 'labels.npy')
        
        with open(self.dataset_dir / 'files.txt') as fp:
            self.all_files = [line.strip() for line in fp]
        
        # Since the backing dataset contains all files, we need to create a map 
        # from the files subset we're given into the full files listing    
        self.index_map = np.zeros(len(files_subset), dtype=np.int64)

        all_file_index = {file:i for i,file in enumerate(self.all_files)}
        for i, file in enumerate(files_subset):
            self.index_map[i] = all_file_index[file.name]

        self.epoch = 0
        self.seen_epoch_indices = set()
        
    def set_transforms(self, transforms):
        pass

    def __len__(self):
        return len(self.index_map)
        
    def __getitem__(self, subset_index):
        # We determine whether we are in a new epoch by looking 
        # whether this index has been used in the current epoch. 
        # If it has we assume we're in a new epoch
        if subset_index in self.seen_epoch_indices:
            self.seen_epoch_indices = set()
            self.epoch += 1
            # If we reach the end of the epochs in the pre-augmented 
            # dataset we cycle back to the start again
            self.epoch %= self.max_epochs
        self.seen_epoch_indices.add(subset_index)
        
        full_index = self.index_map[subset_index]
        data_item = self.data[self.epoch, full_index]
        image = torch.tensor(data_item)
        label = self.labels[full_index]

        return image, label

    
    @classmethod
    def make_cache_dir(cls, cache_directory: Path, dataset: ImageDataset, max_epochs: int):
        from tqdm import tqdm, trange
        
        dataloader = DataLoader(dataset=dataset, batch_size=8, num_workers=4, drop_last=False, shuffle=False)
        dataset_length = len(dataset)

        labels_path = cache_directory / 'labels.npy'
        np.save(labels_path, dataset.labels)
        files_path = cache_directory / 'files.txt'
        with open(files_path, 'w') as fp:
            for p in dataset.images:
                fp.write(f'{p.name}\n')

        cache_directory.mkdir(exist_ok=True, parents=True)
        data_array_path = cache_directory / 'data.npy'
        data_array = None

        for epoch in trange(max_epochs, desc="Preaugmenting Epoch"):
            start_index = 0
            for i, batch in enumerate(tqdm(dataloader, desc="Preaugmenting Batch")):
                x, y = batch
                x = x.numpy()
                batch_size, n_channels, height, width = x.shape
                    
                if data_array is None:
                    shape = (max_epochs, dataset_length, n_channels, height, width)
                    data_array = open_memmap(data_array_path, mode='w+', shape=shape, dtype=x.dtype)

                end = start_index + batch_size
                data_array[epoch, start_index:end] = x
                start_index = end
                
    


def check_jpeg_format(image_path):
    try:
        img = Image.open(image_path)
        return img.format == 'JPEG'
    except (IOError, OSError):
        # If an IOError or OSError is raised, the file is not a valid image
        return False

# Images are structured like Breedname_number.jpg
#All images with 1st letter as captial are cat images
#images with small first letter are dog images
def get_label(file_path: Path):
    # Check if first letter is a capital letter
    if file_path.name[0].isupper():
        label = 'cat'
    else:
        label = 'dog'
    return label

def setup_data_splits():
    dataset = Path("data") / "oxfordiii"
    OxfordIIITPet(dataset, download=True)
    files = list(dataset.glob('**/*.jpg'))
    
    # Some images seem to actually be GIFs, DALI can't handle this so we filter them away
    files = [image for image in files if check_jpeg_format(image)]
    labels = [get_label(f) for f in files]
    
    cache_directory = dataset / 'cache'
    cache_directory.mkdir(exist_ok=True, parents=True)
    
    
    if not (cache_directory / 'data.npy').exists():
        full_dataset = ImageDataset(files, labels, transforms=get_training_transforms())
        PreAugmentedDataset.make_cache_dir(cache_directory, full_dataset, max_epochs=2)

    indices = np.arange(len(files))

    modeling_split_indices, test_indices,  = train_test_split(indices, test_size=0.1, stratify=labels)
    modeling_labels = [labels[i] for i in modeling_split_indices]
    train_indices, dev_indices = train_test_split(modeling_split_indices, test_size=0.1, stratify=modeling_labels)

    test_images = [files[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    dev_images = [files[i] for i in dev_indices]
    dev_labels = [labels[i] for i in dev_indices]
    train_images = [files[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    with open("data_split_indices.json", 'w') as fp:
        # JSON can't serialize Path objects, let's make them strings
        data_splits = dict(test_images=[str(p) for p in test_images], 
                        test_labels=test_labels, 
                        dev_images=[str(p) for p in dev_images], 
                        dev_labels=dev_labels, 
                        train_images=[str(p) for p in train_images], 
                        train_labels=train_labels)
        json.dump(data_splits, fp, sort_keys=True, indent=2)
    
    training_dataset = PreAugmentedDataset(cache_directory, train_images, train_labels)
    dev_dataset = ImageDataset(dev_images, dev_labels)
    test_dataset = ImageDataset(test_images, test_labels)
    
    return training_dataset, dev_dataset, test_dataset


def train_on_dataloader(model, training_dataloader, optimizer, loss_fn, device, iteration=0):
    model.train()
    for training_batch in tqdm(training_dataloader, desc='training batch', leave=False):
        optimizer.zero_grad()
        x, y = training_batch
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        iteration += 1
    return iteration

def evaluate_on_dataloader(model, dataloader, loss_fn, device):
    with torch.no_grad():
        model.eval()
        losses = 0
        n_samples = 0
        logits = []
        probabilities = []
        labels = []
        
        for test_batch in tqdm(dataloader, desc='test batch', leave=False):
            x, y = test_batch
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            probs = torch.sigmoid(prediction)
            logits.append(prediction.cpu().numpy())
            labels.append(y.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            # A minor detail, but since we're not dropping any batches we 
            # can't just take the mean of all the losses over all batches. 
            # This would slightly overweight the last batch if it's smaller than the others.
            batch_n_samples = len(x)
            n_samples += batch_n_samples
            losses += loss.item()*batch_n_samples
            
        mean_test_performance = losses / n_samples
        logits = np.concatenate(logits).flatten()
        labels = np.concatenate(labels).flatten()
        probabilities = np.concatenate(probabilities).flatten()
        roc_auc = roc_auc_score(labels, logits)
        results = {'roc_auc': roc_auc, 'loss': mean_test_performance, 
                   'logits': logits, 'labels': labels, 
                   'probabilities': probabilities}
        return results
        

def train_model(model, training_dataloader, dev_dataloader, optimizer, loss_fn, device, max_epochs, iteration=0):
    best_model = model
    best_roc_auc = float('-inf')

    early_stopping_patience = 3
    epochs_of_no_progress = 0
    for epoch in trange(max_epochs, desc='epoch'):
        iteration = train_on_dataloader(model, training_dataloader, optimizer, loss_fn, device, iteration)
        evaluation_results = evaluate_on_dataloader(model, dev_dataloader, loss_fn, device)
        dev_roc_auc = evaluation_results['roc_auc']
        if dev_roc_auc > best_roc_auc:
            epochs_of_no_progress = 0
            best_roc_auc = dev_roc_auc
            best_model = deepcopy(model)
            model_path = Path('models') / 'best_model.pth'
            model_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_of_no_progress += 1
        
        if epochs_of_no_progress >= early_stopping_patience:
            print("Patience has run out, early stopping")
            break
        
    #### Now train the whole model, using the best one during the frozen training              
    return best_model


def get_training_transforms():
    weights = ResNet18_Weights.DEFAULT
    training_transforms = nn.Sequential(AutoAugment(), weights.transforms())
    return training_transforms
    

def run_training(training_dataset: ImageDataset, dev_dataset: ImageDataset, test_dataset: ImageDataset, device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 1)) # We set the dimension to 1 since we'll use a sigmoid output

    training_transforms = get_training_transforms()
    dev_transforms = weights.transforms()

    training_dataset.set_transforms(training_transforms)
    dev_dataset.set_transforms(dev_transforms)
    test_dataset.set_transforms(dev_transforms)

    training_dataloader = DataLoader(training_dataset, 
                                    batch_size=64, 
                                    drop_last=True, #When training it can actually be a good idea to drop the last batch. If you're using batch normalization somewhere the training can break if there's just a single example in the batch 
                                    shuffle=True,
                                    num_workers=8
                                    )

    dev_dataloader = DataLoader(dev_dataset, 
                                    batch_size=64, 
                                    drop_last=False,
                                    shuffle=False,
                                    num_workers=8
                                    )
    model.to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()

    max_epochs = 1
    for param in model.parameters():
        param.requires_grad = True
    optimizer = AdamW(model.parameters(), lr=0.000001, weight_decay=1e-5)
    model = train_model(model, training_dataloader=training_dataloader, 
                          dev_dataloader=dev_dataloader, optimizer=optimizer, 
                          loss_fn=loss_fn, device=device, max_epochs=max_epochs)
    print("Training is done")
    
    return model


def fit_threshold(targets, predictions):
    # Find the threshold that maximizes the Youden's J statistic (https://en.wikipedia.org/wiki/Youden%27s_J_statistic)
    # See https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/ for a detailed explanation
    # This will essentially pick the threshold which maximizes balanced accuracy
    # In reality you should probably set this depending on how you weight false negatives vs. false positives
    dev_fpr, dev_tpr, dev_threshold = roc_curve(targets, predictions)
    J = dev_tpr - dev_fpr
    ix = np.argmax(J)
    threshold = dev_threshold[ix]
    return threshold


def evaluate_model(model, dataset, tag, device, threshold=None):
    test_dataloader = DataLoader(dataset, 
                                    batch_size=16, 
                                    drop_last=False,
                                    shuffle=False
                                    )

    loss_fn = nn.BCEWithLogitsLoss()
    results = evaluate_on_dataloader(model, test_dataloader, loss_fn, device)
    labels = results['labels']
    logits = results['logits']
    
    if threshold is None:
        threshold = fit_threshold(labels, logits)
        with open(f'{tag}_threshold.txt', 'w') as fp:
            fp.write(f'{threshold}')
    
    class_predictions = results['logits'] >= threshold
    predictions_df = pd.DataFrame(data=dict(files=[str(p) for p in dataset.images],
                                            labels=results['labels'],
                                            logits=results['logits'],
                                            p=results['probabilities'],
                                            class_predictions=class_predictions))
    
    predictions_df.to_csv(f'{tag}_predictions.csv', index=False)
    return threshold
    
        
def main():
    device = torch.device('cuda')
    training_dataset, dev_dataset, test_dataset = setup_data_splits()
    model = run_training(training_dataset, dev_dataset, test_dataset, device=device)
    dev_threshold = evaluate_model(model, dev_dataset, 'dev', device=device)
    evaluate_model(model, test_dataset, 'test', device=device, threshold=dev_threshold)


if __name__ == '__main__':
    main()