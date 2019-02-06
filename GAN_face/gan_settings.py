"""
ПУТИ
0 - МАКС
1 - ДИМА
"""
_PATH_SETTINGS_ = 0


if _PATH_SETTINGS_ == 0:
    _TRUE_PHOTOS_DIR = 'D:\\work_dir\\test_GAN\\LAST_BUILD\\true_frontal_photos\\'
    _TRAIN_IMG_PATH = 'D:\\work_dir\\test_GAN\\LAST_BUILD\\blender_render\\__person_0\\'

    _IMAGES_SAVE_PATH = 'D:\\work_dir\\test_GAN\\LAST_BUILD\\gen_photos\\'

    _GRAPHVIZ_PATH = 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
    _SUMMARY_PATH = 'mobilenet'

elif _PATH_SETTINGS_ == 1:
    _TRUE_PHOTOS_DIR = 'D:\\PycharmProjects\\GAN_faces\\true_frontal_photos\\'
    _TRAIN_IMG_PATH = 'G:\\faces_extended\\img\\__person_0\\'

    _IMAGES_SAVE_PATH = ''

    _GRAPHVIZ_PATH = 'D:\\Graphviz2.38\\bin\\'
    _SUMMARY_PATH = 'mobilenet'


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


# _MOBILENET_INPUT_SHAPE = (224, 224, 3)
_MOBILENET_INPUT_SHAPE = (224, 224, 3)

_INPUT_TENSOR_SHAPE    =  (4, 224, 224, 3)
# _INPUT_TENSOR_SHAPE    =  Input(shape=_INPUT_TENSOR_SHAPE)




"""
НАСТРОЙКИ ОБУЧЕНИЯ
"""
_TRAIN_STEPS = 2000
_BATCH_SIZE = 1
_SAVE_INTERVAL = 1
_OUTPUT_IMAGES_X = 3
_OUTPUT_IMAGES_Y = 3












"""
ИМПОРТИРУЕМЫЕ НАСТРОЙКИ
"""

# РАЗРЕЩЕНИЕ, СТРОКИ  \\  СТОБЦЫ
IMG_RES = (_IMG_ROWS, _IMG_COLS)
