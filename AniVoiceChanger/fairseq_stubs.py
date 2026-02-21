"""
Minimal fairseq stubs to load RVC's hubert_base.pt checkpoint
without the full fairseq library installed.
"""
import sys
import types


class _StubClass:
    """Generic stub that accepts any arguments."""
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"<Stub {self.__class__.__name__}>"


def _create_module(name):
    """Create a stub module and register it in sys.modules."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def install_stubs():
    """Install all the fairseq stubs needed to load hubert_base.pt."""
    # Only install if fairseq is not already available
    if "fairseq" in sys.modules and not isinstance(sys.modules["fairseq"], types.ModuleType):
        return

    # Create the module tree
    modules_to_create = [
        "fairseq",
        "fairseq.data",
        "fairseq.data.data_utils",
        "fairseq.data.dictionary",
        "fairseq.dataclass",
        "fairseq.dataclass.configs",
        "fairseq.dataclass.utils",
        "fairseq.models",
        "fairseq.models.hubert",
        "fairseq.models.hubert.hubert_asr",
        "fairseq.modules",
        "fairseq.modules.grad_multiply",
        "fairseq.tasks",
        "fairseq.tasks.hubert_pretraining",
    ]

    for name in modules_to_create:
        _create_module(name)

    # Create stub classes that the pickle expects

    class Dictionary(_StubClass):
        """Stub for fairseq.data.dictionary.Dictionary"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.symbols = []
            self.count = []
            self.indices = {}

        def __getstate__(self):
            return self.__dict__

        def __setstate__(self, state):
            self.__dict__.update(state)

    class FairseqConfig(_StubClass):
        pass

    class HubertConfig(_StubClass):
        pass

    class HubertCtcConfig(_StubClass):
        pass

    class HubertPretrainingConfig(_StubClass):
        pass

    class HubertPretrainingTask(_StubClass):
        pass

    class GradMultiply(_StubClass):
        pass

    # Assign to modules
    sys.modules["fairseq.data.dictionary"].Dictionary = Dictionary
    sys.modules["fairseq.dataclass.configs"].FairseqConfig = FairseqConfig
    sys.modules["fairseq.models.hubert"].HubertConfig = HubertConfig
    sys.modules["fairseq.models.hubert.hubert_asr"].HubertCtcConfig = HubertCtcConfig
    sys.modules["fairseq.tasks.hubert_pretraining"].HubertPretrainingConfig = HubertPretrainingConfig
    sys.modules["fairseq.tasks.hubert_pretraining"].HubertPretrainingTask = HubertPretrainingTask
    sys.modules["fairseq.modules.grad_multiply"].GradMultiply = GradMultiply
