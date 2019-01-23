

"""
ОСНОВНЫЕ НАСТРОЙКИ
"""
# РАЗРЕЩЕНИЕ, СТРОКИ  \\  СТОБЦЫ
_IMG_ROWS = 600
_IMG_COLS = 600
_CHANNEL = 3

_LIGHTNING = 13
_ROTATION  = 12


_MOBILENET_INPUT_SHAPE = (224, 224, 3)
_INPUT_TENSOR_SHAPE    =  (600, 600, 3, _ROTATION, _LIGHTNING)

_TRAIN_IMG_PATH = 'G:\\faces\\img\\person_0\\'



"""
НАСТРОЙКИ ОБУЧЕНИЯ
"""
_TRAIN_STEPS = 2000
_BATCH_SIZE = 4
_SAVE_INTERVAL = 500
_OUTPUT_IMAGES_X = 3
_OUTPUT_IMAGES_Y = 3
_GENERATED_FACES_PATH = 'D:\\PycharmProjects\\GAN_faces\\generated_faces_mobilenet\\'












"""
ИМПОРТИРУЕМЫЕ НАСТРОЙКИ
"""

# РАЗРЕЩЕНИЕ, СТРОКИ  \\  СТОБЦЫ
IMG_RES = (_IMG_ROWS, _IMG_COLS)
