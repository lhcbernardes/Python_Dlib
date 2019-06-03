import dlib
import cv2
import numpy as np

def imprimePontos(imagem, pontosFacias):
    for p in pontosFacias.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0), 2)

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_5_face_landmarks.dat")
imagem = cv2.imread("fotos/treinamento/ronald.0.1.jpg")
imagemRgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
facesDetectadas = detectorFace(imagemRgb, 0)
facesPontos = dlib.full_object_detections()
for face in facesDetectadas:
    pontos = detectorPontos(imagemRgb, face)
    facesPontos.append(pontos)
    imprimePontos(imagem, pontos)

imagens = dlib.get_face_chips(imagemRgb, facesPontos)
for img in imagens:
    imagemBgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Imagem original", imagem)
    cv2.waitKey(0)
    cv2.imshow("Imagem alinhada", imagemBgr)
    cv2.waitKey(0)

#cv2.imshow("5 pontos", imagem)
#cv2.waitKey(0)
cv2.destroyAllWindows()