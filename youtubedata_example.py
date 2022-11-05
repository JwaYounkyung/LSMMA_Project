### before run this code, you need to fix mtcnn.py code
### line 270 'return faces' -> 'return faces, batch_boxes'

# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

import time
import pafy
import cv2
from PIL import Image, ImageDraw

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

url = 'https://www.youtube.com/watch?v=155cI2v1l-s'
video = pafy.new(url)
best = video.getbest(preftype='mp4') # Selects the stream with the highest resolution

dist_thresh = 1.2

def collate_fn(x):
    return x[0]

# initilaize the embedding list of the comparative group
def initialize(mtcnn, resnet):

    dataset=datasets.ImageFolder('photos') # photos folder path 
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names
    # {0: 'angelina_jolie', 1: 'bradley_cooper', 2: 'kate_siegel', 3: 'paul_rudd', 4: 'shea_whigham', 5: 'taylor_swift'}
    
    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        faces, prob = mtcnn(img, return_prob=True) 

        for face in faces:
            if face is not None and prob>0.90: # if face detected and porbability > 90%
                emb = resnet(face.unsqueeze(0).to(device)) # passing cropped face into resnet model to get embedding matrix
                embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
                name_list.append(idx_to_class[idx]) # names are stored in a list

    data = [embedding_list, name_list]
    torch.save(data, 'data.pt') # saving data.pt file


# draw the rectangular and name in video
def extract_face_info(img, box, name, min_dist):
    (x1, y1, x2, y2) = box.tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2) # draw face rectangle

    if min_dist < dist_thresh:
        cv2.putText(img, "Face : " + name, (x1, y2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        # cv2.putText(img, "Dist : " + str(min_dist), (x1, y2 + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    else:
        cv2.putText(img, 'No matching faces', (x1, y2 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


# face verification
def recognize_face(img, face, database, network):

    embedding_list = database[0] # getting embedding data
    name_list = database[1] # getting list of names
    
    dist_list = [] # list of matched distances, minimum distance is used to identify the person

    emb = network(face.unsqueeze(0).to(device)).detach() # detech is to make required gradient false

    # photos에 있는 사진 중에 가장 비슷한 이미지가 무엇인지 판별
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    
    name = name_list[idx_min]
    min_dist = min(dist_list)

    return name, min_dist

real_sec = [8,9,10,11,17,18,19,20,21,24,28,29,30,31,32,33,34,35,36,37,38,41,42,43,44,45,46,59,60,61,67,68,69,72,73,76,77,111,112,118,119,120,124,125,130,131,136,137,138,139,141,142,143,153,154,156,157,158,159,160,177,178,188,189,190]
real_frame = list()
captured_frame = list()

def recognize(member):
    # model define
    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20, keep_all=True, device=device) # initializing mtcnn for face detection
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) # initializing resnet for face img to embeding conversion

    initialize(mtcnn, resnet)
    cap=cv2.VideoCapture(best.url) 

    # fps
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    
    fps = round(fps)
    print(fps)

    saved_data = torch.load('data.pt')
    
    frame = 0

    # frame 단위
    while True:
        frame += 1
        ret, img = cap.read() # ret 정상적으로 읽어 왔는지

        if ret: 
            # face detection
            faces, boxes = mtcnn(img) # returns cropped face and bounding box
            
            if (faces != None): # 영상에서 얼굴이 잡히지 않을 수도 있음
                for i, face in enumerate(faces):
                    name, min_dist = recognize_face(img, face, saved_data, resnet)
                    # extract_face_info(img, boxes[i], name, min_dist)

                    if(name == member):
                        captured_frame.append(frame)
                        extract_face_info(img, boxes[i], name, min_dist)

            cv2.imshow('Recognizing faces', img) 
            
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break


    # real_sec to real_frame
    for i in real_sec: 
        for j in range(fps): # 30 fps
            real_frame.append(i*fps+j)
                
    sum = 0
        
    for i in real_frame: 
        if i in captured_frame: 
            sum += 1  

    print("Accuracy: {:.2f} % \n".format(((float(sum))/(len(real_frame)))*100))   
    
    cap.release()
    cv2.destroyAllWindows()


recognize('eunji')