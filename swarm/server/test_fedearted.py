import torch
import pytest
from swarm.server.federated import agglomerate

def test_mean_agglomerat():
    model_one = torch.nn.Linear(1, 1)
    model_two = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model_one.weight.fill_(2)
        model_two.weight.fill_(4)
    model_one_weights = model_one.state_dict()
    model_two_weights = model_two.state_dict()
    agglomerate_weight = agglomerate([model_one_weights, model_two_weights])
    assert agglomerate_weight["weight"] == 3


def test_mean_agglomerat_sequential():
    model_one = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Linear(100, 1000),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1000, 1)
    )
    model_two = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.Linear(100, 1000),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1000, 1)
    )
    model_one_weights = model_one.state_dict()
    model_two_weights = model_two.state_dict()
    agglomerate_weight = agglomerate([model_one_weights, model_two_weights])
    assert agglomerate_weight["0.weight"][0] != model_one_weights["0.weight"][0]
    assert agglomerate_weight["0.weight"][0] != model_two_weights["0.weight"][0]

@pytest.mark.xfail
def test_mean_agglomerat_lazy_conv():
    model_one = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.LazyConv1d(10, 3),
    )
    model_two = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.LazyConv1d(10, 3),
    )
    model_one_weights = model_one.state_dict()
    model_two_weights = model_two.state_dict()
    agglomerate_weight = agglomerate([model_one_weights, model_two_weights])
    assert agglomerate_weight["0.weight"][0] != model_one_weights["0.weight"][0]
    assert agglomerate_weight["0.weight"][0] != model_two_weights["0.weight"][0]


def test_mean_agglomerat_conv_2d():
    model_one = torch.nn.Conv2d(28*28, 5, 3)
    model_two = torch.nn.Conv2d(28*28, 5, 3)
    model_one_weights = model_one.state_dict()
    model_two_weights = model_two.state_dict()
    agglomerate_weight = agglomerate([model_one_weights, model_two_weights])
    assert agglomerate_weight["weight"][0][0][0][0] != model_one_weights["weight"][0][0][0][0]
    assert agglomerate_weight["weight"][0][0][0][0] != model_two_weights["weight"][0][0][0][0]