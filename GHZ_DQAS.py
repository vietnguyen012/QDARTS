
from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
    Iterable,
    Dict,
)
import sys 
from DQASsearch import DQAS_search
from utils import set_op_pool
from vag import GHZ_vag 
from pennylane import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

ghz_pool = [
    ("RY", 0),
    ("RY", 1),
    ("RY", 2),
    ("CNOT", 0, 1),
    ("CNOT", 1, 0),
    ("CNOT", 0, 2),
    ("CNOT", 2, 0),
    ("Hadamard", 0),
    ("Hadamard", 1),
    ("Hadamard", 2)]

set_op_pool(ghz_pool)
c = len(ghz_pool)
p = 5

stp, nnp, history= DQAS_search(
    GHZ_vag,
    nq=3,
    p=p,
    batch=1,
    epochs=1000,
    verbose=True,
    nnp_initial_value=np.zeros([p, c]),
    structure_opt=tf.keras.optimizers.Adam(learning_rate=0.15),
)
plt.plot(history)
plt.savefig('/home/fptu/QDARTS/loss.png', bbox_inches='tight')
