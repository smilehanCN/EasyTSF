from easytsf.model.registry import get_model_class


def test_arrow_model_is_registered_without_importing_optional_runtime_dependencies():
    model_cls = get_model_class("ARROW")
    assert model_cls.__name__ == "Model"
