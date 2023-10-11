from models.RNN import _RNN_Base, nn

class Model(_RNN_Base):
    _cell = nn.LSTM