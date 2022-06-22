from tensorflow.keras.optimizers import *

def get_optimizer(cfg):
    if cfg.train.optimizer_name == "sgd":
        return SGD(learning_rate=cfg.train.lr, momentum=0.9, nesterov=True)
    elif cfg.train.optimizer_name == "adam":
        return Adam(learning_rate=cfg.train.lr)
    elif cfg.train.optimizer_name == "RMSprop":
        return RMSprop(learning_rate=cfg.train.lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")