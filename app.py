import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import os



feature_list = np.array(pickle.load(open('embeddings4.pkl','rb')))
filenames = pickle.load(open('filenames34.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])







# st.title('Customer Adviser')

option = st.sidebar.selectbox(
        'Choose your favourite BRANDS',
     ('WELCOME','ADDIDAS', 'AMERICAN TOURISTER','LAKME','LEVIS','NIVEA','PUMA'))


def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

def NIVEA():
    original = Image.open('BrandImages/Nivea_logo.svg (1).png')
    st.image(original)
    st.title(option)

    image2 = cv2.imread('BrandImages/NIVEA.png')
    features = feature_extraction(os.path.join('BrandImages/NIVEA.png'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(filenames[indices[0][1]])
        st.caption('ARJUN RAMPAL')

    with col2:
        st.image(filenames[indices[0][11]])
        st.caption('TEEJAY SIDHU')


def WELCOME():
    # st.caption('customer adviser')
    st.image('BrandImages/BGround.png')



def PUMA():
    original = Image.open('BrandImages/PUMA.png')
    st.title(option)
    st.image(original)

    image2 = cv2.imread('BrandImages/PUMA.png')
    features = feature_extraction(os.path.join('BrandImages/PUMA.png'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(filenames[indices[0][1]])
        st.caption('VIRAT KOHLI')

    with col2:
        st.image(filenames[indices[0][0]])
        st.caption('K.L RAHUL')

    with col3:
        st.image(filenames[indices[0][4]])
        st.caption('BANI J')




def LAKME():
    original = Image.open('BrandImages/LAKME.png')
    st.image(original)
    st.header(option)

    image2 = cv2.imread('BrandImages/LAKME.png')
    features = feature_extraction(os.path.join('BrandImages/LAKME.png'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2, col3, col4, col5 = st.columns(5)





    with col1:
        st.image(filenames[indices[0][1]])
        st.caption('DIA MIRZA')


    with col2:
        st.image(filenames[indices[0][11]])
        st.caption('ARPITA MEHTA')





def AT():
    original = Image.open('BrandImages/AmericanTourister.jpg')
    st.image(original)
    st.header(option)

    image2 = cv2.imread('BrandImages/AmericanTourister.jpg')
    features = feature_extraction(os.path.join('BrandImages/AmericanTourister.jpg'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(filenames[indices[0][0]])
        st.caption('RONALDO')



def ADDIDAS():
    original = Image.open('BrandImages/Addidas.jpg')
    st.image(original)
    st.title(option)


    image2 = cv2.imread('BrandImages/Addidas.jpg')
    features = feature_extraction(os.path.join('BrandImages/Addidas.jpg'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:
        st.image(filenames[indices[0][5]])
        st.caption('HIMA DAS')



def LEVI():
    original = Image.open('BrandImages/LEVI.png')
    st.image(original)
    st.title("LEVI'S")

    image2 = cv2.imread('BrandImages/LEVI.png')
    features = feature_extraction(os.path.join('BrandImages/LEVI.png'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2 = st.columns(2)


    with col1:
        st.image(filenames[indices[0][4]])
        st.caption('ZOYA AKHTAR')

    with col2:
        st.image(filenames[indices[0][5]])
        st.caption('HARSHVARDHAN KAPOOR')






if option == 'WELCOME':
    WELCOME()
if option == 'NIVEA':
    NIVEA()

if option=='LAKME':
    LAKME()

if option =='PUMA':
    PUMA()

if option =='ADDIDAS':
    ADDIDAS()




if option=='AMERICAN TOURISTER':
    AT()


if option=='LEVIS':
    LEVI()











