"""
ПУТИ
0 - МАКС
1 - ДИМА
"""
_PATH_SETTINGS_ = 1



if _PATH_SETTINGS_ == 0:
    _TRUE_PHOTOS_DIR = 'D:\\work_dir\\test_GAN\\LAST_BUILD\\true_frontal_photos\\'
    _TRAIN_IMG_PATH = 'D:\\work_dir\\test_GAN\\LAST_BUILD\\blender_render\\__person_0\\'
    _BLENDER_PHOTOS_DIR = ''

elif _PATH_SETTINGS_ == 1:
    # _TRUE_PHOTOS_DIR = 'D:\\PycharmProjects\\GAN_faces\\true_frontal_photos\\'
    _TRUE_PHOTOS_DIR = 'G:\\faces\\img\\person_0\\'
    _TRAIN_IMG_PATH = 'G:\\faces\\img\\person_0\\'
    _BLENDER_PHOTOS_DIR = ''






"""
ОСНОВНЫЕ НАСТРОЙКИ
"""
# РАЗРЕЩЕНИЕ, СТРОКИ  \\  СТОБЦЫ
_IMG_ROWS = 600
_IMG_COLS = 600
_CHANNEL = 3

_PERSONS = 7

_LIGHTNING = 13
_ROTATION  = 120


_MOBILENET_INPUT_SHAPE = (224, 224, 3)
_INPUT_TENSOR_SHAPE    =  (4, 224, 224, 3)
# _INPUT_TENSOR_SHAPE    =  Input(shape=_INPUT_TENSOR_SHAPE)




"""
НАСТРОЙКИ ОБУЧЕНИЯ
"""
_TRAIN_STEPS = 2000
_BATCH_SIZE = 1
_SAVE_INTERVAL = 500
_OUTPUT_IMAGES_X = 3
_OUTPUT_IMAGES_Y = 3
_GENERATED_FACES_PATH = 'D:\\work_dir\\test_GAN\\generated_faces_mobilenet\\'












"""
ИМПОРТИРУЕМЫЕ НАСТРОЙКИ
"""

# РАЗРЕЩЕНИЕ, СТРОКИ  \\  СТОБЦЫ
IMG_RES = (_IMG_ROWS, _IMG_COLS)
