import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def load_weights(model, weights):
    """
    Loads the weights of only the layers present in the given model.
    """
    pretrained_dict = torch.load(weights, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def compute_classweights(dataset, num_classes):
    if num_classes == 1:
        print("Binary classification, no class_weights")
        return None

    print(f"Number of classes: {num_classes}")
    all_labels = []
    for data in dataset:
        # image, label - don't need image
        _, label = data
        all_labels.append(label)  # All the BIRADS labels
        # print(type(all_labels))

    all_labels = np.array(all_labels)
    # 0 1 2 3 4
    # print(type(all_labels))
    unique_classes = np.unique(all_labels)
    balanced_weight = compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=all_labels
    )

    return balanced_weight
