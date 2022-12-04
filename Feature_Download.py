from SoccerNet.Downloader import SoccerNetDownloader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', default="data/features", help='feature download directory')
parser.add_argument("--pca", action="store_true", default=True)
args = parser.parse_args()

downloader = SoccerNetDownloader(LocalDirectory=args.dir)

if args.pca==False:
    downloader.downloadGames(files=["Labels-cameras.json", "Labels-v2.json", "1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train", "valid", "test"], verbose=True)
else:
    downloader.downloadGames(files=["Labels-cameras.json", "Labels-v2.json", "1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train", "valid", "test"], verbose=True)
