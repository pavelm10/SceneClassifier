import keras


class ModelSaver(keras.callbacks.Callback):
    def __init__(self, model_dir, model_name, **kwargs):
        super(ModelSaver, self).__init__(**kwargs)
        self.epoch_accuracy = {} # loss at given epoch
        self.epoch_loss = {} # accuracy at given epoch
        self.model_dir = model_dir
        self.model_name = model_name
    def on_epoch_begin(self,epoch, logs={}):
        # Things done on beginning of epoch.
        return

    def on_epoch_end(self, epoch, logs={}):
        # things done on end of the epoch
        tmpl = f"{self.model_name}_{epoch}.h5"
        path = self.model_dir / tmpl
        self.epoch_accuracy[epoch] = logs.get("acc")
        self.epoch_loss[epoch] = logs.get("loss")
        self.model.save(path.as_posix()) # save the model