import pytest

from torchrush.module.auto import AutoRush
from torchrush.module.lenet5 import LeNet


@pytest.fixture(scope="module")
def model_fixture():
    input_size = [1, 28, 28]
    embedding_size = 79
    return LeNet(input_size=input_size, embedding_size=embedding_size)


def test_save_pretrained_load_from_pretrained(tmp_path, model_fixture):
    model_fixture.save_pretrained(tmp_path)

    input_size = (1, 28, 28)
    embedding_size = 79
    # load exported custom rush model
    model_from_checkpoint = LeNet(input_size=input_size, embedding_size=embedding_size)
    model_from_checkpoint = model_from_checkpoint.from_pretrained(tmp_path.as_posix())

    assert model_from_checkpoint.input_size == model_fixture.input_size
    assert model_from_checkpoint.fc2.out_features == model_fixture.fc2.out_features
    assert model_from_checkpoint.fc2.out_features == embedding_size
    assert model_from_checkpoint.fc2.in_features == model_fixture.fc2.in_features


def test_autorush(tmp_path, model_fixture):
    model_fixture.save_pretrained(tmp_path)

    # load exported custom rush model via autorush
    model_from_checkpoint = AutoRush.from_pretrained(tmp_path.as_posix())

    assert model_from_checkpoint.input_size == list(model_fixture.input_size)
    assert model_from_checkpoint.fc2.out_features == model_fixture.fc2.out_features
    assert model_from_checkpoint.fc2.out_features == 79
    assert model_from_checkpoint.fc2.in_features == model_fixture.fc2.in_features
