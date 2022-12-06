import json
from datetime import datetime, timedelta

class create_timestamp():
    def __init__(self, json_file_dir) -> None: #gets json file
        f = open(json_file_dir)
        self.f_data = json.load(f)

    def make_stamp(self, which_json, cnt_goal): #extracts information from json file -> returns list
        if which_json == 'action':
            all_pred = []
            for i in self.f_data['predictions']:
                label = i['label']
                confidence = float(i['confidence'])
                time_stamp = i['gameTime']
                if confidence >= 0.5 and label == 'Goal':
                    all_pred.append([label, confidence, time_stamp])
            
            total_cnt = cnt_goal + 2
            if len(all_pred) > total_cnt: 
              all_pred = all_pred[:total_cnt]
              
        elif which_json == 'replay':
            all_pred = []
            for i in self.f_data['predictions']:
                label = i['label']
                confidence = float(i['confidence'])
                time_stamp = i['gameTime']
                if confidence >= 0.95 and label == 'replay':
                    all_pred.append([label, confidence, time_stamp])
        
        return all_pred

class create_start_end(): #create start and end points for each clip
  def __init__(self, time_list, first_start_second, second_start_second):
    self.time_list = time_list
    self.first_start = first_start_second
    self.second_start = second_start_second

  def create_datetime(self): #changes the timestamps from the json file to real timestamps in the original video
    timestamp_list = []
    for i in self.time_list:
      i_time_info = i[2]
      i_list = i_time_info.split(" - ")
      i_half = i_list[0]
      i_timestamp = datetime.strptime(i_list[1], "%M:%S")
      # print(i_timestamp)
      if i_half == '1':
        i_realtime = i_timestamp + timedelta(seconds=self.first_start)
      elif i_half == '2':
        i_realtime = i_timestamp + timedelta(seconds=self.second_start)
      i_starttime = i_realtime - timedelta(seconds=10)
      i_endtime = i_realtime + timedelta(seconds=10)
      # print(i_starttime)
      # print(i_endtime)
      # print() 
      timestamp_list.append([i[0], i[1], i_starttime, i_endtime])
    
    timestamp_list = sorted(timestamp_list, key=lambda x: x[2]) #clips are sorted in chronological order
    #print(timestamp_list)
    return timestamp_list

  def create_string(self, datetime_list): #change the timestamps from datetime to string
    string_list = []
    for i in datetime_list:
      start_string = i[2].strftime("%H:%M:%S")
      end_string = i[3].strftime("%H:%M:%S")
      string_list.append([i[0], i[1], start_string, end_string])
    return string_list

  def no_overlap(self, str_list): #clips may have overlap -> change so that there is no overlap
    no_overlap_list = str_list
    for i in range(len(no_overlap_list)-1):
      if no_overlap_list[i][3] > no_overlap_list[i+1][2]:
        no_overlap_list[i][3] = no_overlap_list[i+1][2]
    return no_overlap_list

#https://stackoverflow.com/questions/3766633/how-to-sort-with-lambda-in-python
#https://bobbyhadz.com/blog/python-add-seconds-to-datetime#:~:text=Use%20the%20timedelta()%20class,of%20seconds%20to%20the%20datetime.
#https://www.geeksforgeeks.org/read-json-file-using-python/
 

if __name__ == '__main__':
  action_info = create_timestamp('/content/drive/MyDrive/LSMMA_Project/SoccerNetv2-DevKit/Task1-ActionSpotting/TemporallyAwarePooling/src/models/NetVLAD++_reproduce/outputs_test/soccermatch/results_spotting.json')
  action_list = action_info.make_stamp('action', 6)
  #print(action_list)
  for i in action_list:
    print(i)
  # replay_info  = create_timestamp('/content/drive/MyDrive/LSMMA_Project/HighlightClassification/FineTuning/models/NetVLAD++_highlight/outputs_test/soccermatch/results_spotting.json')
  # replay_list = replay_info.make_stamp('replay')
  # #print(replay_list)
  # replay_list.extend(action_list)
  # replay_start_end = create_start_end(replay_list, 319, 3470)
  # replay_start_end_list = replay_start_end.create_datetime()
  # #print(replay_start_end_list)
  # replay_string = replay_start_end.create_string(replay_start_end_list)
  # #print(replay_string)
  # for i in replay_string:
  #   print(i)
  # print("replay_string:", len(replay_string))

  # no_ol = replay_start_end.no_overlap(replay_string)
  # for i in no_ol:
  #   print(i)
  # print("no_overlap: ", len(no_ol))
            