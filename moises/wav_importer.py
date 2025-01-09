import matplotlib.pyplot as plt
import soundfile as sf
import argparse
import requests
from io import BytesIO

def parse():
    parser = argparse.ArgumentParser(usage="python3 wav_importer.py [file url]", 
                                     description="Cosa c'è da capire")
    parser.add_argument("url", type=str, help="riporta l'url del file da leggere")
    return parser.parse_args()

def main():
    arg = parse()
    byteData = requests.get(arg.url)
    byteData.raise_for_status()
    audio, sample_rate = sf.read(BytesIO(byteData.content))

    plt.plot(audio)
    plt.show()
    
    file = arg.url.split("/")[-1]
    sf.write(file, audio, sample_rate)
    
    plt.plot(sf.read(file)[0])
    plt.show()
    

if __name__ == "__main__":
    main()