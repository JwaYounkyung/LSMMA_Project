
from pytube import YouTube


class download_video():
    def __init__(self, url, file_title, chosen_res, output_dir) -> None: 
        self.url = url #video url
        self.file_title = file_title #name of the saved video file
        self.chosen_res = chosen_res #resolution
        self.output_dir = output_dir #directory where the output is to be saved

    def download_url(self): 
        vid = YouTube(self.url)
        vid.streams.filter(progressive=True, file_extension='mp4', res=self.chosen_res).first().download(output_path=self.output_dir, filename=self.file_title)
        #https://stackoverflow.com/questions/47649536/python-download-youtube-with-specific-filename
        #https://pytube.io/en/latest/api.html#pytube.Stream.download
        #https://www.geeksforgeeks.org/pytube-python-library-download-youtube-videos/
        #https://readthedocs.org/projects/python-pytube/downloads/pdf/stable/
        #https://github.com/AdamSpannbauer/youtube_reaction_face/issues/1

        print(self.file_title + " is downloaded")



if __name__=='__main__':
    dl = download_video('https://www.youtube.com/watch?v=xVWwDc5aNDM', 'test_vid.mp4', '360p', '/content/drive/MyDrive')
    dl.download_url()