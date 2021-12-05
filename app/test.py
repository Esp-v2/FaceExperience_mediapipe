import streamlit as st
from PIL import Image,ImageOps
import numpy as np
import mediapipe as mp
import tensorflow as tf
import cv2

# 画像を正方形にする
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
# mediapipeで点群取得する
def img_to_mediapipe(targetImg):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        
        #image=cv2.imread(targetImg)
        #results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image=targetImg
        results = face_mesh.process(targetImg)
        
        if not results.multi_face_landmarks:
            facemesh_csv = []
            col_label = []
            for xyz in range(468):
                col_label.append(str(xyz) + "_x")
                col_label.append(str(xyz) + "_y")
                col_label.append(str(xyz) + "_z")
            for _ in range(3):
                facemesh_csv.append(np.nan)
      
        # mediapipeで推定可能な画像に対して、3次元カラムと推定3次元座標をlistに追加
        else:
            annotated_image = image.copy()
            
            for face_landmarks in results.multi_face_landmarks:
                facemesh_csv = []
                col_label = []

                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
            
            for xyz, landmark in enumerate(face_landmarks.landmark):
                col_label.append(str(xyz) + "_x")
                col_label.append(str(xyz) + "_y")
                col_label.append(str(xyz) + "_z")
                facemesh_csv.append(landmark.x)
                facemesh_csv.append(landmark.y)
                facemesh_csv.append(landmark.z)
                
            st.image(annotated_image)
    return facemesh_csv

def main():
    # タイトル表示
    st.title('表情識別')
    st.write('## 読み込んだ画像の人物の表情を識別します')

    # file upload
    uploaded_file = st.file_uploader('Choose a image file')
    if uploaded_file is not None:
        # 画像を読み込む
        uploaded_img = Image.open(uploaded_file)
        uploaded_img = ImageOps.exif_transpose(uploaded_img)  # 画像を適切な向きに補正する
        st.image(uploaded_img)

        # 画像を48x48の正方形にする
        uploaded_img=expand2square(uploaded_img,(0, 0, 0))
        uploaded_img = uploaded_img.resize((48,48))
        st.image(uploaded_img)
        
        st.info(uploaded_file)
        
        # mediapipeで点群を取得する
        img_array=np.array(img_to_mediapipe(uploaded_img.convert('RGB')))

        # 判定
        model = tf.keras.models.load_model(r"C:\Users\proje\Desktop\NN_model.hdf5")
        pred = model.predict_classes(img_array.reshape(1,-1),batch_size=1,verbose=0)

        # 結果表示
        st.info(pred)




if __name__ == "__main__":
    main()