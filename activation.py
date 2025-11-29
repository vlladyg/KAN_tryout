import math

def relu(x, get_derivative=False):

    return x*(x>0) if not get_derivative else 1.0*(x>=0)

def sigmoid_act(x, get_derivative=False):
    if not get_derivative:
        return 1.0/(1.0 + math.exp(-x))
    else:
        return sigmoid(x) * (1.0 - sigmoid(x))

def tanh_act(x, get_derivative=False):
    if not get_derivative:
        return math.tanh(x)
    else:
        return 1.0 - math.tanh(x)**2