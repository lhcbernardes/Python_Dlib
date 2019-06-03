import cv2
import dlib

subdetector = ["Olhar a frente", "Vista a esquerda", "Vista a direita",
               "A frente girando a esquerda", "A frente girando a direita"]

imagem = cv2.imread("fotos/grupo.0.jpg")
detector = dlib.get_frontal_face_detector()
facesDetectadas, pontuacao, idx = detector.run(imagem, 1, -2)
#print(facesDetectadas)
#print(pontuacao)
#print(idx)
for i, d in enumerate(facesDetectadas):
    #print(i)
    #print(d)
    print("Detecção: {}, pontuação: {}, Sub-detector: {}".format(d, pontuacao[i], subdetector[idx[i]]))
    e, t, d, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
    cv2.rectangle(imagem, (e, t), (d, b), (0, 0, 255), 2)

cv2.imshow("Detector hog", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()