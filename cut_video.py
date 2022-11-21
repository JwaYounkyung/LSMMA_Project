from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.fx.all import *


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
        self.text_clips_start = []
        self.text_clips_end = []
        self.save_dir_point = save_dir + "/" + final_vid_title

        org_vid = VideoFileClip(self.vid_dir)
        front_duration = 0
        
        for i in range(len(self.start_time)):
          part_start, part_end = self.turn_second(self.start_time[i], self.end_time[i])
          if i == 0:
            part_clip = org_vid.subclip(part_start, part_end)
            self.text_clips_start.append(front_duration)
            front_duration += part_end - part_start
            self.text_clips_end.append(front_duration)
          else:
            part_clip = org_vid.subclip(part_start, part_end).set_start(self.text_clips_end[i-1]-1).crossfadein(1.0)
            self.text_clips_start.append(front_duration)
            front_duration += part_end - part_start-1
            self.text_clips_end.append(front_duration)
          final_clips.append(part_clip)
          ###
          

        final_vid = CompositeVideoClip(final_clips)
        final_vid.write_videofile(self.save_dir_point)
    
    def create_caption(self, captions,  save_dir, caption_video_title):
        caption_list = []
        text_feature = lambda txt: TextClip(txt, font='Arial', fontsize=40, color='white')
        for i in range(len(self.text_clips_start)):
          caption_list.append(((self.text_clips_start[i], self.text_clips_end[i]), captions[i]))
        caption_vid = SubtitlesClip(caption_list, text_feature)
        background_vid = VideoFileClip(self.save_dir_point)
        
        background_caption = CompositeVideoClip([background_vid, caption_vid.set_pos((0.1, 0.8), relative=True)] )
        background_caption.write_videofile(save_dir + "/" + caption_video_title)

    

#https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html?highlight=videofileclip#moviepy.video.io.VideoFileClip.VideoFileClip
#https://www.geeksforgeeks.org/moviepy-concatenating-multiple-video-files/
#https://www.geeksforgeeks.org/moviepy-saving-video-file-clip/
#https://zulko.github.io/moviepy/index.html (the moviepy github in general)
#https://stackoverflow.com/questions/66363548/trimming-batch-of-videos-using-timestamp
#https://www.geeksforgeeks.org/moviepy-getting-cut-out-of-video-file-clip/
#https://www.geeksforgeeks.org/moviepy-inserting-text-in-the-video/
#https://blog.dennisokeeffe.com/blog/2021-07-31-annotating-video-clips-with-moviepy
#https://www.geeksforgeeks.org/moviepy-composite-video-adding-cross-fade-in-effect/
#https://stackoverflow.com/questions/72285020/how-to-add-transitions-between-clips-in-moviepy
#https://stackoverflow.com/questions/36667702/adding-subtitles-to-a-movie-using-moviepy



if __name__ == "__main__":
    start_time = ['00:00:00', '00:01:00', '00:02:00']
    end_time = ['00:00:07', '00:01:10','00:02:10']
    captions = ['cake1 and ', 'cake2', 'cake3']
    cutv = cut_vid(start_time, end_time,  '/content/drive/MyDrive/test_vid.mp4' )
    cutv.final_cut_vid('/content/drive/MyDrive', "final_vid.mp4")
    cutv.create_caption(captions, '/content/drive/MyDrive', 'captioned_video.mp4')
