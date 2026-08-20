from config import load_config, print_config


CONFIG_PATH = "configs/unsupervised_bert_base.yaml"


def main():
    config = load_config(CONFIG_PATH)

    print_config(config)

    assert config["experiment_name"] == (
        "unsupervised_bert_base"
    )

    assert config["model_name"] == (
        "bert-base-uncased"
    )

    assert config["batch_size"] == 64

    assert config["learning_rate"] == 3.0e-5

    assert config["temperature"] == 0.05

    assert config["epochs"] == 1

    assert config["max_seq_length"] == 32

    assert config["mlp_only_train"] is True

    print("\nConfiguration test: PASSED")


if __name__ == "__main__":
    main()