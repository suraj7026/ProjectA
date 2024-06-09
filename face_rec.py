import os
import pandas as pd
import cv2
import numpy as np
import re
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import redis


# Face Recognition Model 
faceapp = FaceAnalysis(name = 'buffalo_l',
                       root = 'InsightFaceModels',
                       providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0,det_size=(640,640),det_thresh = 0.5)


# Redis Connection

hostname = 'redis-13055.c305.ap-south-1-1.ec2.cloud.redislabs.com'
port = 13055
password = 'fOz4pYOutbMZi3dHLk1bhfBSmMws6spu'

r = redis.StrictRedis(host = hostname,
                      port = port,
                      password = password)

# Get data
def get_data(name = 'academy:register'):
    name = 'academy:register'
    retrieve_dict = r.hgetall(name = name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype = np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(),index))
    retrieve_series.index = index
    retrieve_df = retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ["Name_Roll","Features"]
    retrieve_df[['Name','Roll']] = retrieve_df['Name_Roll'].apply(lambda x:x.split('@')).apply(pd.Series)
    return retrieve_df[['Name','Roll','Features']]


# Search 

def multiple_face_search(dataframe, feature_column, test_vector, threshold=0.4, name_role=['Name', 'Roll']):
    
    dataframe = dataframe.copy()

    X_list = dataframe[feature_column].tolist()
    X = np.asarray(X_list)

    similarity = pairwise.cosine_similarity(X, test_vector.reshape(1, -1))
    similar_arr = np.array(similarity).flatten()
    dataframe["cosine"] = similar_arr

    data_filter = dataframe.query(f"cosine > {threshold}")
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter["cosine"].argmax()
        name, role = data_filter.loc[argmax][name_role]
    else:
        name = "Unknown"
        role = "Unknown"
    return name, role


def face_recognition(image,dataframe, feature_column, threshold=0.4, name_role=['Name', 'Roll']):
        
    face_results = faceapp.get(image)
    test_copy = image.copy()
    
    for r in face_results:
        x1, y1, x2, y2 = r['bbox'].astype(int)
        embeddings = r['embedding']  
        name, role = multiple_face_search(dataframe,
                                           feature_column,
                                           test_vector=embeddings,
                                           name_role = name_role,
                                           threshold=0.5)
        
        if name == "Unknown":
            color = (0,0,255)
        else:
            color = (0,255,0)
        cv2.rectangle(test_copy, (x1, y1), (x2, y2), color, 2)
        
        
        text_gen = f"{name}, {role}"
        cv2.putText(test_copy, text_gen, (x1, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    return test_copy

def face_recognition_upload_image(image, dataframe, feature_column, threshold=0.4, name_role=['Name', 'Roll']):
    face_results = faceapp.get(image)
    
    names = []
    rolls = []
        
    for r in face_results:
        
        embeddings = r['embedding']
        name, role = multiple_face_search(dataframe, feature_column, test_vector=embeddings, name_role=name_role, threshold=0.5)
            
        if name != "Unknown":
            names.append(name)
            rolls.append(role)
            
       
        
    return names, rolls
    
