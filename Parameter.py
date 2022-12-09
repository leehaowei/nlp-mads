from dataclasses import dataclass


@dataclass
class M1Param:
    max_features: int
    min_df: int
    n_estimators: int


@dataclass
class LSTMParam:
    input_len: int
    lstm_size: int
    embedding_size: int
    dropout_rate: float
    activation_out: str
    leraning_rate: float
    loss: str
    metric: str
    batch: int
    epoch: int
    verbose: int


@dataclass
class LSTM2Param(LSTMParam):
    recurrent_dropout: float


@dataclass
class M4Param:
    text_col: str
    label: str
    max_features: int
    max_len: int
    verbose: 1
    max_expoch: int
