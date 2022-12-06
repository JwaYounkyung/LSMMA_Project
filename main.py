import argparse
import os
from get_video import download_video
from cut_video import cut_vid, make_second
from make_timestamp import create_timestamp, create_start_end
from datetime import datetime


def inference_action(): #makes json file for action spotting 
  os.chdir('SoccerNetv2-DevKit/Task1-ActionSpotting/TemporallyAwarePooling/src')
  os.system('python main_inference_for_youtube.py --SoccerNet_path=../../../../results --version 2 --feature_dim=512 --model_name=NetVLAD++_reproduce --batch_size 32 --test_only')

def inference_replay(curr_dir): #makes json file for highlight classification 
  os.chdir(curr_dir)
  os.chdir('HighlightClassification/FineTuning')
  os.system('python main.py --SoccerNet_path=../../results/ \
--max_epochs 1000 \
--test_only \
--version 5 \
--features=ResNET_TF2_PCA512.npy \
--feature_dim=512 \
--model_name=NetVLAD++_highlight_nlayer2 \
--batch_size 32 \
--evaluation_frequency 10 \
--n_layers 2')




def get_timestamp(curr_dir, first_half_time, second_half_time, num_goal): #gets timestamps(starting point, ending point) for each clip, including information for captions
  
  action_info = create_timestamp(curr_dir + '/SoccerNetv2-DevKit/Task1-ActionSpotting/TemporallyAwarePooling/src/models/NetVLAD++_reproduce/outputs_test/soccermatch/results_spotting.json')
  action_list = action_info.make_stamp('action', num_goal)
  replay_info  = create_timestamp(curr_dir + '/HighlightClassification/FineTuning/models/NetVLAD++_highlight/outputs_test/soccermatch/results_spotting.json')
  replay_list = replay_info.make_stamp('replay', num_goal)
  replay_list.extend(action_list)

  replay_start_end = create_start_end(replay_list, first_half_time, second_half_time)
  replay_start_end_list = replay_start_end.create_datetime()

  replay_string = replay_start_end.create_string(replay_start_end_list)
  no_ol = replay_start_end.no_overlap(replay_string)

  return no_ol

def make_cut_input(timestamp_info): #organizes the information so they can be used to cut the original video
  start_timestamp = []
  end_timestamp = []
  caption = []

  cnt = 1
  for i in timestamp_info: 
    start_timestamp.append(i[2])
    end_timestamp.append(i[3])
    caption.append("Type: " + i[0] + "   No : " + str(cnt) + "   Confidence: " + str(i[1])[:5])
    cnt += 1

  return start_timestamp, end_timestamp, caption

#referenced: https://github.com/KevinQian97/11755-ISR-HW1/blob/main/test_mlp.py
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("get_vid_url")
  parser.add_argument("curr_dir")
  parser.add_argument("get_vid_dir")
  parser.add_argument("cut_vid_dir")
  parser.add_argument("cut_vid_name")
  parser.add_argument("first_half")
  parser.add_argument("first_half_end")
  parser.add_argument("second_half")
  parser.add_argument("second_half_end")
  parser.add_argument("number_of_goals")

  return parser.parse_args()



if __name__ == '__main__':
  args = parse_args()
  get_url = args.get_vid_url
  curr_dir = args.curr_dir
  get_dir = args.get_vid_dir
  cut_dir = args.cut_vid_dir
  cut_name = args.cut_vid_name
  start_first = args.first_half
  end_first = args.first_half_end
  start_second = args.second_half
  end_second = args.second_half_end  
  num_goals = args.number_of_goals

  start_first_cl= make_second(start_first)
  first_sec = start_first_cl.timestamp_second()

  start_second_cl = make_second(start_second)
  second_sec = start_second_cl.timestamp_second()

  end_first_cl = make_second(end_first)
  first_end_sec = end_first_cl.timestamp_second()

  end_second_cl = make_second(end_second)
  second_end_sec = end_second_cl.timestamp_second()

  first_duration = first_end_sec - first_sec #duration of first half 
  second_duration = second_end_sec - second_sec #duration of second half
  
  print("first duration: ", first_duration)
  print("second duration: ", second_duration)

  
  # print(get_url) 
  # dl_low = download_video(get_url, "low_res.mp4", "360p", curr_dir+get_dir)
  # dl_high = download_video(get_url, "high_res.mp4", "720p", curr_dir+get_dir)
  # dl_low.download_url()
  # dl_high.download_url()

  
  ###******feature extraction******###
  first_feature = 'soccermatch_1 '
  second_feature = 'soccermatch_2 '

  cmd_1 = curr_dir + '/SoccerNetv2-DevKit/Features/VideoFeatureExtractor.py  --path_video='
  cmd_path = curr_dir + '/video/original/low_res.mp4 ' 
  cmd_2 = '--path_features=' 
  cmd_path_2 = curr_dir + '/results/'
  cmd_3 = '--start='
  cmd_duration = ' --duration='
  cmd_4 = ' --PCA='
  cmd_path_3 = curr_dir + '/SoccerNetv2-DevKit/Features/pca_512_TF2.pkl'
  cmd_5 =' --PCA_scaler='
  cmd_path_4 = curr_dir + '/SoccerNetv2-DevKit/Features/average_512_TF2.pkl'
  os.system('python ' + cmd_1 + cmd_path + cmd_2 + cmd_path_2 + first_feature + cmd_3 + str(first_sec) + cmd_duration + str(first_duration) + cmd_4 + cmd_path_3 + cmd_5 + cmd_path_4)
  os.system('python ' + cmd_1 + cmd_path + cmd_2 + cmd_path_2 + second_feature + cmd_3 + str(second_sec) + cmd_duration + str(second_duration) + cmd_4 + cmd_path_3 + cmd_5 + cmd_path_4)
  ###******feature extraction*******###


  ####******inference action******####
  inference_action() 
  ###******inference action******####



  ###*******inference replay******###
  inference_replay(curr_dir)
  ##*******inference replay******###



  ###*******make timestamps******###
  timestamp_list = get_timestamp(curr_dir, first_sec, second_sec, num_goals)
  #print(timestamp_list)
  #print(len(timestamp_list))
  ###*******make timestamps******###



  start_time, end_time, captions = make_cut_input(timestamp_list)


  cut_video =  cut_vid(start_time, end_time, curr_dir + get_dir + "/" + "high_res.mp4")
  cut_video.final_cut_vid(curr_dir + cut_dir, cut_name + ".mp4")
  cut_video.create_caption(captions, curr_dir + cut_dir, "captioned_" + cut_name + ".mp4")
    

#https://www.programiz.com/python-programming/datetime/strftime
#https://stackoverflow.com/questions/431684/equivalent-of-shell-cd-command-to-change-the-working-directory
#https://stackabuse.com/executing-shell-commands-with-python/
#https://stackoverflow.com/questions/62883924/how-to-convert-a-string-x-hours-x-minutes-x-seconds-to-datetime-in-python

