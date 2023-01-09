import pytest
import torch
from tests import _PATH_DATA

from data import mnist

dataset = mnist()
train = dataset[0]
testing = dataset[1]
N_train = 25000
N_test = 5000
labels = []
for i in train:
  labels.append(i[1].item())
assert len(train) == N_train and len(testing) == N_test, "Dataset did not have the correct number of samples"
assert train[0][0].shape == (28,28), "Input tensor did not have the correct shape"
assert len(set(labels)) == 10, "Not all labels are represented in the sample"
# print(train[0])
#assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
#assert that all labels are represented

# import os.path
# @pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
# def test_something_about_data():
#    ...

@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 54)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected