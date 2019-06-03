import cv2
import dlib

#imagem = cv2.imread("fotos/grupo.0.jpg")
#imagem = cv2.imread("fotos/grupo.1.jpg")
#imagem = cv2.imread("fotos/grupo.2.jpg")
#imagem = cv2.imread("fotos/grupo.3.jpg")
#imagem = cv2.imread("fotos/grupo.4.jpg")
#imagem = cv2.imread("fotos/grupo.5.jpg")
#imagem = cv2.imread("fotos/grupo.6.jpg")
imagem = cv2.imread("fotos/grupo.7.jpg")

detectorHog = dlib.get_frontal_face_detector()
facesDetectadasHog, pontuacao, idx = detectorHog.run(imagem, 2)

detectorCNN = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat")
facesDetectadasCNN = detectorCNN(imagem, 2)

for i, d in enumerate(facesDetectadasHog):
    print(pontuacao[i])
print("")
for face in facesDetectadasCNN:
    print(face.confidence)

