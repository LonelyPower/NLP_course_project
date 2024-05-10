import os
import random
import json


def read_split_data(root: str, sample_rate: float, seed: int):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    random.seed(seed)
    train_data_path, train_label = read_and_sample_single_folder(os.path.join(root, "train"), sample_rate)
    val_data_path, val_label = read_and_sample_single_folder(os.path.join(root, "val"), sample_rate)
    return train_data_path, train_label, val_data_path, val_label


def read_and_sample_single_folder(root: str, sample_rate: float):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    text_class = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    text_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(text_class))
    with open('class_indices.json', 'w', encoding='utf-8') as json_file:
        json.dump(class_indices, json_file, indent=4)
        
    all_text_path = []
    all_text_label = []
    for cls in text_class:
        cls_path = os.path.join(root, cls)
        texts = [os.path.join(cls_path, text) for text in os.listdir(cls_path) if text.endswith('.txt')]
        all_text_path.extend(texts)
        all_text_label.extend([class_indices[cls]] * len(texts))
    print("{} texts were found in the dataset {} before sampling.".format(len(all_text_path), root))

    n_samples = int(len(all_text_path) * sample_rate)
    sampled_indices = random.sample(range(len(all_text_path)), n_samples)
    sampled_data_path = [all_text_path[i] for i in sampled_indices]
    sampled_labels = [all_text_label[i] for i in sampled_indices]
    print("{} texts were sampled from the dataset {}.".format(len(sampled_data_path), root))
    
    return sampled_data_path, sampled_labels


if __name__ == '__main__':
    root_folder = "data"  
    train_data_path, train_label, val_data_path, val_label = read_split_data(root_folder)
