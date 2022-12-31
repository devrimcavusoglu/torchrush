from torchrush.module.lenet5 import LeNet


def test_save_pretrained_load_from_pretrained():
    # create and export custom rush model
    save_dir = "tests/data/custom_export/"

    embedding_size = 79
    lenet = LeNet(input_size=(1, 28, 28), embedding_size=embedding_size)
    lenet.save_pretrained(save_dir)

    # load exported custom rush model
    lenet2 = LeNet(input_size=(1, 28, 28), embedding_size=embedding_size)
    lenet2 = lenet.from_pretrained(save_dir)

    assert lenet2.input_size == list(lenet.input_size)
    assert lenet2.fc2.out_features == lenet.fc2.out_features
    assert lenet2.fc2.out_features == embedding_size
    assert lenet2.fc2.in_features == lenet.fc2.in_features


def test_autorush():
    from torchrush.module.auto import AutoRush

    # create and export custom rush model
    save_dir = "tests/data/custom_export/"
    input_size = [1, 28, 28]

    embedding_size = 79
    lenet = LeNet(input_size=input_size, embedding_size=embedding_size)
    lenet.save_pretrained(save_dir)

    # load exported custom rush model via autorush
    rushmodule = AutoRush.from_pretrained(save_dir)

    assert rushmodule.input_size == list(lenet.input_size)
    assert rushmodule.fc2.out_features == lenet.fc2.out_features
    assert rushmodule.fc2.out_features == embedding_size
    assert rushmodule.fc2.in_features == lenet.fc2.in_features
