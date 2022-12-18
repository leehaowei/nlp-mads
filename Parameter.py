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
    max_epoch: int
    learning_rate: float
    n_cycles: int
    cycle_len: int
    cycle_mult: int


@dataclass
class BertParam:
    text_col: str
    label_col: str
    max_features: int
    max_len: int
    verbose: int
    pre_trained: str
    batch: int
    max_epoch: int
    n_cycles: int
    cycle_len: int
    cycle_mult: int
