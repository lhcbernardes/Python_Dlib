import dlib
import glob
import cv2
import os

opcoes = dlib.simple_object_detector_training_options()
opcoes.add_left_right_image_flips = True
opcoes.C = 5
#dlib.train_simple_object_detector("recursos/treinamento_delirium.xml", "recursos/detector_delirium.svm", opcoes)

detector = dlib.simple_object_detector("recursos/detector_delirium.svm")
for imagem in glob.glob(os.path.join("delirium", "*.jpg")):
    img = cv2.imread(imagem)
    objetosDetectados = detector(img, 2)
    for d in objetosDetectados:
        e, t, d, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
        cv2.rectangle(img, (e,t), (d, b), (0,0,255), 2)

    cv2.imshow("Detector Delirium", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()