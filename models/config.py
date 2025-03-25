
from enum import Enum
import argparse
from data.dataloader import STEADDataset
class MODE_TYPE(Enum):
    TRAIN          = 1
    PREDICT        = 2
    ALL            = 3

class MODEL_TYPE(Enum):
    WGANOMALY = 1
    EFFIGANOMALY = 2
    ANOGAN = 3

class Config:
    def __init__(self):
        self.MODE               = MODE_TYPE.TRAIN

        self.MODEL_TYPE         = MODEL_TYPE.WGANOMALY
        self.ORIGINAL_DB_FILE   = "/Users/user/Desktop/Temp/waveforms.hdf5"
        self.METADATA_PATH      = "data/metadata.csv"
        self.MODEL_FILE_NAME    = "models/model_default.pt"
        self.MODEL_PATH         = "models/"

        # Below parameters are used in extract_db script to extract certain window in database
        self.DATASET_FILE  = "data/waveforms_4s_new_full.hdf5" 
        self.CSV_FILE      = "data/model_details.csv"
        self.ORIGINAL_SAMPLING_RATE = 100 
        self.TRAINING_WINDOW        = 60 # in seconds
        self.BASE_SAMPLING_RATE     = 100
        # Optional: e.g., torchvision.transforms.Compose([...])
        self.DATASET                = STEADDataset(file=self.DATASET_FILE, csv_file=self.CSV_FILE,transform=None) 

        self.TEST_DATA              = self.DATASET.get_test_data()
        self.TRAIN_DATA             = self.DATASET.get_train_data()

        # Calculated parameters
        self.SAMPLE_WINDOW_SIZE = self.BASE_SAMPLING_RATE * self.TRAINING_WINDOW


        #self.TEST_DATA_SPLIT_RATIO = 0.8
        #self.IS_SPLIT_DATA         = True

class WGanomalyConfig:
    def __init__(self):
        self.input_size = 6000, 
        self.input_channels =3, 
        self.base_channels  =8, 
        self.kernel_size = 7, 
        self.stride = 4, 
        self.padding = 3, 
        self.alpha = 0.2, 
        self.latent_dim = 100, 
        self.shuffle_factor = 2, 
        self.num_gpus=1, 
        self.num_extra_layers=0, 
        self.add_final_conv=True

class NNCFG:
    def __init__(self):
        self.learning_rate          = 0.001
        self.epoch_count            = 2
        self.batch_size             = 32

        self.adam_beta1             = 0.9
        self.adam_beta2             = 0.999
        self.adam_gamma             = 0.1

        self.detection_threshold    = 0.5

        # Dynamic variables
        self.training_loss          = None
        self.optimizer              = None
        self.model_id               = None



    def argParser(self, cfg: Config):
        parser = argparse.ArgumentParser()

        # Add arguments
        parser.add_argument('--learning_rate', type=float, help='Learning rate of the NN (int)')
        parser.add_argument('--epoch_count', type=int, help='Number of epoches')
        parser.add_argument('--batch_size', type=int, help='Batch size')

        parser.add_argument('--adam_beta1', type=float, help='Beta 1 of Adam optimizer')
        parser.add_argument('--adam_beta2', type=float, help='Beta 2 of Adam optimizer')
        parser.add_argument('--adam_gamma', type=float, help='Gamma of Adam optimizer')
        parser.add_argument('--model_file_name', type=str, help='Path to the model file')
        parser.add_argument('--detection_threshold', type=float, help='Detection threshold of when one output neuron exist')

        args = parser.parse_args()

        self.learning_rate   = args.learning_rate   if args.learning_rate is not None else self.learning_rate
        self.epoch_count     = args.epoch_count     if args.epoch_count is not None else self.epoch_count
        self.batch_size      = args.batch_size      if args.batch_size is not None else self.batch_size

        self.adam_beta1     = args.adam_beta1 if args.adam_beta1 is not None else self.adam_beta1
        self.adam_beta2     = args.adam_beta2 if args.adam_beta2 is not None else self.adam_beta2
        self.adam_gamma     = args.adam_gamma if args.adam_gamma is not None else self.adam_gamma

        if args.model_file_name:
            cfg.MODEL_FILE_NAME = cfg.MODEL_PATH + args.model_file_name + ".pt"

        self.detection_threshold = args.detection_threshold if args.detection_threshold is not None else self.detection_threshold

        mode_messages = {
            MODE_TYPE.TRAIN: f"Training Hyperparameters: Learning Rate = {self.learning_rate}, Epoch count = {self.epoch_count}, Batch Size = {self.batch_size}",
            MODE_TYPE.PREDICT: f"Detection Threshold = {self.detection_threshold}",
            MODE_TYPE.ALL: f"Training Hyperparameters: Learning Rate = {self.learning_rate}, Epoch count = {self.epoch_count}, Batch Size = {self.batch_size}. Testing Hyperparameter: Detection Threshold = {self.detection_threshold}"
            }
        
        print(mode_messages.get(cfg.MODE, "Invalid Mode"))
