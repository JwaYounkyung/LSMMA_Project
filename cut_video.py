from moviepy.editor import *

class cut_vid():
    def __init__(self, start_time, end_time, vid_dir) -> None:
        self.vid_dir = vid_dir #directory where the original video is stored 
        self.start_time = start_time #list of starting timestamp (each element: str)
        self.end_time = end_time  #list of ending timestamp (each element: str)

    def turn_second(self, start_point, end_point):
        start_h, start_m, start_s = start_point.split(":")
        end_h, end_m, end_s = end_point.split(":")

        start_h = int(start_h)
        start_m = int(start_m)
        start_s = int(start_s)
        end_h = int(end_h)
        end_m = int(end_m)
        end_s = int(end_s)

        start_second = start_h*3600 + start_m*60 + start_s
        end_second = end_h*3600 + end_m*60 + end_s

        return start_second, end_second
    
    def final_cut_vid(self, save_dir, final_vid_title): 
        #save_dir: where the final video is to be saved
        #final_vid_title: name of the final video
        final_clips = []
        org_vid = VideoFileClip(self.vid_dir)
        for i in range(len(self.start_time)):
          part_start, part_end = self.turn_second(self.start_time[i], self.end_time[i])
          part_clip = org_vid.subclip(part_start, part_end)
          final_clips.append(part_clip)

        final_vid = concatenate_videoclips(final_clips)
        final_vid.write_videofile(save_dir + "/" + final_vid_title)
        


#https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html?highlight=videofileclip#moviepy.video.io.VideoFileClip.VideoFileClip
#https://www.geeksforgeeks.org/moviepy-concatenating-multiple-video-files/
#https://www.geeksforgeeks.org/moviepy-saving-video-file-clip/
#https://zulko.github.io/moviepy/index.html (the moviepy github in general)
#https://stackoverflow.com/questions/66363548/trimming-batch-of-videos-using-timestamp
#https://www.geeksforgeeks.org/moviepy-getting-cut-out-of-video-file-clip/

if __name__ == "__main__":
    start_time = ['00:00:00', '00:01:00']
    end_time = ['00:00:10', '00:01:10']
    cutv = cut_vid(start_time, end_time,'/content/drive/MyDrive/test_vid.mp4' )
    cutv.final_cut_vid('/content/drive/MyDrive', "final_vid.mp4")



