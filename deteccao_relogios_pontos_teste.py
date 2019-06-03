import dlib
import cv2
import glob
import os

detectorRelogio = dlib.simple_object_detector("recursos/detector_relogios.svm")
detectorPontosRelogio = dlib.shape_predictor("recursos/detector_relogios_pontos.dat")

print(dlib.test_shape_predictor("recursos/teste_relogios_pontos.xml", "recursos/detector_relogios_pontos.dat"))

def imprimirPontos(imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0))

for arquivo in glob.glob(os.path.join("relogios_teste", "*.jpg")):
    imagem = cv2.imread(arquivo)
    objetosDetectados = detectorRelogio(imagem, 2)
    for relogio in objetosDetectados:
        e, t, d, b = (int(relogio.left()), int(relogio.top()), int(relogio.right()), int(relogio.bottom()))
        cv2.rectangle(imagem, (e, t), (d, b), (0, 0, 255), 2)
        pontos = detectorPontosRelogio(imagem, relogio)
        imprimirPontos(imagem, pontos)

    cv2.imshow("Detector pontos", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()