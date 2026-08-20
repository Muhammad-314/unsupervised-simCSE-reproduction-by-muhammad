from pathlib import Path

from experiment import create_experiment_directory


CONFIG_PATH = (
    "configs/unsupervised_bert_base.yaml"
)


def main():
    experiment_dir = create_experiment_directory(
        experiment_name="test_experiment",
        config_path=CONFIG_PATH,
    )

    print("Created experiment directory:")
    print(experiment_dir)

    assert experiment_dir.exists()

    copied_config = (
        experiment_dir / "config.yaml"
    )

    assert copied_config.exists()

    print("\nExperiment directory test: PASSED")


if __name__ == "__main__":
    main()