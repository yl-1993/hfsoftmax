from .resnet import *
from .hynet import *
from .classifier import Classifier, HFClassifier, HNSWClassifier
from .ext_layers import ParameterClient


samplerClassifier = {
    'hf': HFClassifier,
    'hnsw': HNSWClassifier,
}
