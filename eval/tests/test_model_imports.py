import importlib
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))


def _clear_eval_model_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]


class ModelImportTests(unittest.TestCase):
    def tearDown(self) -> None:
        _clear_eval_model_modules()

    def test_videomolmo_import_does_not_require_llava_st_environment(self) -> None:
        _clear_eval_model_modules()

        module = importlib.import_module("models.videomolmo")

        self.assertTrue(hasattr(module, "VideoMolmoModel"))

    def test_models_package_lazily_loads_videomolmo_without_llava_st_environment(self) -> None:
        _clear_eval_model_modules()

        import models

        self.assertTrue(hasattr(models, "VideoMolmoModel"))

    def test_llava_st_import_does_not_mutate_python_path(self) -> None:
        _clear_eval_model_modules()

        original_path = list(sys.path)
        module = importlib.import_module("models.llava_st")

        self.assertEqual(sys.path, original_path)

    def test_llava_st_dependency_loader_falls_back_to_bundled_inference_source(self) -> None:
        _clear_eval_model_modules()

        temp_dir = Path(tempfile.mkdtemp())
        utils_path = temp_dir / "inference" / "src" / "utils.py"
        utils_path.parent.mkdir(parents=True)
        utils_path.write_text(
            textwrap.dedent(
                """
                def get_variables(conversations):
                    return conversations, {"loaded_from": "bundled-source"}
                """
            ),
            encoding="utf-8",
        )

        fake_llava = types.ModuleType("llava")
        fake_llava.__file__ = str(temp_dir / "llava" / "__init__.py")

        fake_conversation = types.ModuleType("llava.conversation")
        fake_conversation.__file__ = str(temp_dir / "llava" / "conversation.py")

        fake_constants = types.ModuleType("llava.constants")
        fake_constants.DEFAULT_IM_END_TOKEN = "</im>"
        fake_constants.DEFAULT_IM_START_TOKEN = "<im>"
        fake_constants.DEFAULT_IMAGE_PATCH_TOKEN = "<image_patch>"
        fake_constants.DEFAULT_IMAGE_TOKEN = "<image>"
        fake_constants.DEFAULT_SLOW_VID_END_TOKEN = "</slow_vid>"
        fake_constants.DEFAULT_SLOW_VID_START_TOKEN = "<slow_vid>"
        fake_constants.DEFAULT_VID_END_TOKEN = "</vid>"
        fake_constants.DEFAULT_VID_START_TOKEN = "<vid>"
        fake_constants.DEFAULT_VIDEO_PATCH_TOKEN = "<video_patch>"
        fake_constants.DEFAULT_VIDEO_TOKEN = "<video>"
        fake_constants.IGNORE_INDEX = -100
        fake_constants.IMAGE_TOKEN_INDEX = 42

        fake_model = types.ModuleType("llava.model")
        fake_builder = types.ModuleType("llava.model.builder")
        fake_builder.load_lora_model = lambda *args, **kwargs: ("loaded", args, kwargs)

        injected_modules = {
            "llava": fake_llava,
            "llava.conversation": fake_conversation,
            "llava.constants": fake_constants,
            "llava.model": fake_model,
            "llava.model.builder": fake_builder,
        }
        original_modules = {name: sys.modules.get(name) for name in injected_modules}
        sys.modules.update(injected_modules)
        self.addCleanup(self._restore_modules, original_modules)

        module = importlib.import_module("models.llava_st")
        module.LLAVA_ST_DEPENDENCIES_LOADED = False
        original_root_resolver = module._bundled_llava_st_root
        module._bundled_llava_st_root = lambda: temp_dir
        self.addCleanup(setattr, module, "_bundled_llava_st_root", original_root_resolver)

        original_path = list(sys.path)
        module._ensure_llava_st_dependencies()

        self.assertEqual(sys.path, original_path)
        _, variables = module.get_variables([{"value": "x"}])
        self.assertEqual(variables["loaded_from"], "bundled-source")

    def test_videomlomo_alias_is_recognized(self) -> None:
        os.environ.pop("VIDEOMOLMO_REPO", None)
        _clear_eval_model_modules()

        from main import _build_model

        with self.assertRaisesRegex(RuntimeError, "VIDEOMOLMO_REPO is required"):
            _build_model("videomlomo", "videomolmo", 1, 16, 1024, 0.0, 1, 0.9)

    def _restore_modules(self, original_modules) -> None:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
