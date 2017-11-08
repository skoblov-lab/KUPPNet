from os import environ


def load_model(file_path, hparams, device):
    environ["CUDA_VISIBLE_DEVICES"] = device
    from keras import models
    model = models.load_model(file_path)
    return model


if __name__ == '__main__':
    raise RuntimeError
