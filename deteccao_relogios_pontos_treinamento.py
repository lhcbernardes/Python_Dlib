import dlib

opcoes = dlib.shape_predictor_training_options()
dlib.train_shape_predictor("recursos/treinamento_relogios_pontos.xml", "recursos/detector_relogios_pontos.dat", opcoes)