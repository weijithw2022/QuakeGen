from config import Config, MODE_TYPE
from data.train import train

def main():
    cfg = Config()
    if cfg.MODE == MODE_TYPE.TRAIN:
        train(cfg)
    else:
        print("I am at the start")

if __name__ == 'main':
    main()

