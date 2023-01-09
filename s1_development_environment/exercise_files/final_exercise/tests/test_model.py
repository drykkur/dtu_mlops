from tests import _PATH_DATA
from data import mnist
from model import MyAwesomeModel
import torch
import pytest

# def test_error_on_wrong_shape(): cache
#    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#       model(torch.randn(1,2,3))

dataset = mnist()
testing = dataset[1]

model = MyAwesomeModel()
model.eval()
testloader = torch.utils.data.DataLoader(testing, batch_size=1,
                                         shuffle=False, num_workers=4)
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images.float())
        break
assert outputs.shape == (1,10)



