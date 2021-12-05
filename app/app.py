# -*- coding: utf-8 -*-
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image,ImageOps
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def mediapipe_static(dir_input):

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    results = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5).process(cv2.cvtColor(np.array(dir_input), cv2.COLOR_BGR2RGB))

    # mediapipeで推定不可の画像に対して、3次元カラムとNaN要素をlistに追加
    if not results.multi_face_landmarks:
        facemesh_csv = []
        col_label = []
        for xyz in range(468):
            col_label.append(str(xyz) + "_x")
            col_label.append(str(xyz) + "_y")
            col_label.append(str(xyz) + "_z")
        for _ in range(3):
            facemesh_csv.append(np.nan)
    else:
        annotated_image = np.array(dir_input)
        #annotated_image = cv2.cvtColor(np.array(dir_input.copy()), cv2.COLOR_BGR2RGB)
        
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        
        facemesh_csv = []
        col_label = []
        
        for xyz, landmark in enumerate(face_landmarks.landmark):
            col_label.append(str(xyz) + "_x")
            col_label.append(str(xyz) + "_y")
            col_label.append(str(xyz) + "_z")
            facemesh_csv.append(landmark.x)
            facemesh_csv.append(landmark.y)
            facemesh_csv.append(landmark.z)
    
    data = pd.DataFrame([facemesh_csv], columns=col_label)

    return data, annotated_image


def main():
    # タイトル表示
    st.title('表情識別')
    st.write('## 読み込んだ画像の人物の表情を識別します')

    # file upload
    uploaded_file = st.file_uploader('Choose a image file')
    if uploaded_file is not None:
        # 画像を読み込む
        uploaded_img = Image.open(uploaded_file)
        st.image(uploaded_img)
        
        # mediapipeで点群を取得する
        df,meshimg=mediapipe_static(uploaded_img)
        st.image(meshimg)

        # 判定
        model = tf.keras.models.load_model(r"C:\Users\proje\Desktop\NN_model.hdf5")
        img_array=np.array(df.iloc[:, :]).reshape(len(df.index), len(df.columns), 1)
        pred = model.predict_classes(img_array.reshape(1,-1),batch_size=1,verbose=0)

        # 結果表示
        st.info(pred)




if __name__ == "__main__":
    main()