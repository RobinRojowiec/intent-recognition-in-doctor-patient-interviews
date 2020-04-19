import configparser


class Configuration:
    def __init__(self, file_name: str, section="DEFAULT"):
        self.file_name = file_name
        self.config = configparser.ConfigParser()
        self.section = section
        self.load()

    def get_string(self, key):
        return self.config[self.section][key]

    def get_float(self, key):
        return float(self.get_string(key))

    def get_int(self, key):
        return int(self.get_string(key))

    def get_boolean(self, key):
        return self.get_string(key) == 'True'

    def set(self, key, value):
        self.config[self.section][key] = str(value)

    def reset(self):
        for parameter, value in ModelParameter.__dict__:
            self.config[self.section][parameter] = 0

    def load(self):
        self.config.clear()
        self.config.read(self.file_name)

    def save(self):
        with open(self.file_name, 'w+') as configfile:
            self.config.write(configfile)


class ModelParameter:
    EPOCHS = "EPOCHS"
    BATCH_SIZE = "BATCH_SIZE"
    TRAINING_FILE = "TRAINING_FILE"
    TRAINING_SIZE = "TRAINING_SIZE"
    DEV_FILE = "DEV_FILE"
    DEV_SIZE = "DEV_SIZE"
    TEST_FILE = "TEST_FILE"
    LEARNING_RATE = "LEARNING_RATE"
    WEIGHT_DECAY = "WEIGHT_DECAY"
    HIDDEN_SIZE = "HIDDEN_SIZE"
    EMBEDDING_SIZE = "EMBEDDING_SIZE"
    MARGIN = "MARGIN"
    MAX_LENGTH = "MAX_LENGTH"
    OPTIMIZER = "OPTIMIZER"
    CHANNELS = "CHANNELS"
    KERNEL_SIZE = "KERNEL_SIZE"
    USE_PRETRAINED_EMBEDDINGS = "USE_PRETRAINED_EMBEDDINGS"
    PRETRAINED_MODEL = "PRETRAINED_MODEL"


if __name__ == '__main__':
    print(ModelParameter.BATCH_SIZE)
    print(str(ModelParameter))
