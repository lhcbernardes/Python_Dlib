import cv2

imagem = cv2.imread("fotos/grupo.0.jpg")
classificador = cv2.CascadeClassifier("recursos/haarcascade_frontalface_default.xml")
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.2, minSize=(50,50))
print(facesDetectadas)
print("Faces detectadas: ", len(facesDetectadas))
for (x, y, l, a) in facesDetectadas:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

cv2.imshow("Detector haar", imagem)
cv2.waitKey(0)

cv2.destroyAllWindows()