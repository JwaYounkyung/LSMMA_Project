import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="data/features")
downloader.downloadGames(files=["Labels-cameras.json", "Labels-v2.json", "1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train", "valid", "test"], verbose=True)