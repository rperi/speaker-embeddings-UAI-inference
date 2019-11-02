from . import mnist_rot
from . import german
from . import thin_resnet_uai
from . import xvector_uai

DATASETS_DICT = {
    'mnist_rot': mnist_rot,
    'german': german,
    'thin_resnet_uai': thin_resnet_uai,
    'simple_CNN_uai': thin_resnet_uai,
    'thin_CNN_uai': thin_resnet_uai,
    'xvector_uai': xvector_uai,
    'xvector_uai_voices': xvector_uai,
    'xvector_uai_combined': xvector_uai
}
