# Reference from https://github.com/clovaai/aasist/blob/main/download_dataset.py
import os

if __name__ == "__main__":
    cmd = "curl -o ./data/asvspoof2019/LA.zip -# https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip\?sequence\=3\&isAllowed\=y"
    os.system(cmd)
    cmd = "unzip ./data/asvspoof2019/LA.zip -d ./data/asvspoof2019/"
    os.system(cmd)
