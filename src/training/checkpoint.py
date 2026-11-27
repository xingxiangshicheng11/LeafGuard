import torch


def save_checkpoint(path, model, num_classes, class_to_idx, model_name, input_size, config,
                    best_val_macro_f1, optimizer=None, epoch=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "model_name": model_name,
        "input_size": input_size,
        "config": config,
        "best_val_macro_f1": float(best_val_macro_f1),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    torch.save(payload, path)


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)
