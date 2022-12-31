if __name__ == "__main__":
    from torchrush.module.lenet5 import LeNet

    # create and export custom rush model
    save_dir = "examples/custom"
    lenet = LeNet(input_size=(1, 28, 28), embedding_size=84)
    lenet.save_pretrained(save_dir)

    # load exported custom rush model
    lenet = LeNet(input_size=(1, 28, 28), embedding_size=84)
    lenet = lenet.from_pretrained(save_dir)

    # load in auto rush style
    from torchrush.module.auto import AutoRush

    rushmodule = AutoRush.from_pretrained(save_dir)
