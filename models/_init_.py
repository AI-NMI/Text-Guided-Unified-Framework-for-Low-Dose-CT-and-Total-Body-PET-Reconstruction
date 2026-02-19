from .basic_template import TrainTask
from .tuf import corediff, tuf

model_dict = {
    'tuf': tuf,
    'corediff': corediff,
}
