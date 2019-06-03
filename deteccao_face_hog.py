import cv2
import dlib

imagem = cv2.imread("fotos/grupo.0.jpg")
detector = dlib.get_frontal_face_detector()
facesDetectadas = detector(imagem, 3)
print(facesDetectadas)
print("Faces detectadas: ", len(facesDetectadas))
for face in facesDetectadas:
    #print(face)
    #print(face.left())
    #print(face.top())
    #print(face.right())
    #print(face.bottom())
    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)

cv2.imshow("Detector hog", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()