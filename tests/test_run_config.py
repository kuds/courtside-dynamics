"""Tests for TOML run-configuration files (docs/run_config_file_spec.md)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from courtside_dynamics.recipes import RECIPES, build_train_config
from courtside_dynamics.run_config import load_run_config


def _write(tmp_path, text, name="run.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestLoader:
    def test_parses_tables_sentinels_and_provenance(self, tmp_path):
        text = (
            "[train]\n"
            'normalize_reward = "none"\n'
            "early_stop_patience = 20\n"
            "[train.model_kwargs]\n"
            'ent_coef = "auto_0.05"\n'
            "[train.phase_labels]\n"
            '0 = "await_bounce"\n'
            "[env]\n"
            "serve_vy_max = 1.4\n"
            'early_touch_penalty = "NONE"\n'
            "[eval_env]\n"
            "episode_len = 5000\n"
        )
        path = _write(tmp_path, text)
        config = load_run_config(path)
        assert config.train["normalize_reward"] is None
        assert config.train["early_stop_patience"] == 20
        assert config.train["model_kwargs"] == {"ent_coef": "auto_0.05"}
        # TOML keys are strings; the env's phase ids are ints.
        assert config.train["phase_labels"] == {0: "await_bounce"}
        assert config.env == {
            "serve_vy_max": 1.4,
            "early_touch_penalty": None,
        }
        assert config.eval_env == {"episode_len": 5000}
        assert config.text == text
        assert len(config.sha256) == 64
        assert config.raw["train"]["normalize_reward"] == "none"

    def test_unknown_table_is_rejected(self, tmp_path):
        path = _write(tmp_path, "[model]\nent_coef = 0.2\n")
        with pytest.raises(ValueError, match=r"unknown table.*model"):
            load_run_config(path)

    def test_unknown_train_key_suggests_the_real_field(self, tmp_path):
        path = _write(tmp_path, "[train]\nearly_stop_patence = 20\n")
        with pytest.raises(
            ValueError, match=r"early_stop_patence.*early_stop_patience"
        ):
            load_run_config(path)

    def test_code_only_train_keys_are_rejected(self, tmp_path):
        for key in ("env_fn", "extra_callbacks", "warm_start"):
            path = _write(tmp_path, f"[train]\n{key} = 1\n", f"{key}.toml")
            with pytest.raises(ValueError, match=key):
                load_run_config(path)

    def test_builder_owned_train_keys_are_rejected(self, tmp_path):
        # These are written into the kwargs before the file merges, so a
        # file value would silently beat the explicit argument and
        # falsify provenance (an algo="SAC" run recorded as PPO,
        # artifacts redirected away from the mandatory log_dir).
        for key, value in (
            ("algo", '"PPO"'),
            ("log_dir", '"/elsewhere"'),
            ("recipe_name", '"WallBall"'),
        ):
            path = _write(
                tmp_path, f"[train]\n{key} = {value}\n", f"{key}.toml"
            )
            with pytest.raises(ValueError, match=key):
                load_run_config(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_run_config(tmp_path / "absent.toml")

    def test_malformed_toml_names_the_file(self, tmp_path):
        path = _write(tmp_path, "[train\n")
        with pytest.raises(ValueError, match="run.toml"):
            load_run_config(path)

    def test_non_utf8_file_names_the_file(self, tmp_path):
        path = tmp_path / "run.toml"
        path.write_bytes(b"[train]\nn_envs = \xff\n")
        with pytest.raises(ValueError, match=r"run\.toml.*UTF-8"):
            load_run_config(path)

    def test_quoted_boolean_strings_are_rejected(self, tmp_path):
        # 'normalize_reward = "false"' is a truthy Python string that
        # would silently enable the feature the file tried to disable.
        cases = {
            "train": '[train]\nnormalize_reward = "false"\n',
            "model_kwargs": '[train.model_kwargs]\nuse_sde = "true"\n',
            "env": '[env]\nfragment_resets = "False"\n',
            "eval_env": '[eval_env]\nfragment_resets = "TRUE"\n',
            "gate_stage": (
                "[train.performance_gate]\n"
                'metric_key = "bounce_count_ep_mean"\n'
                "threshold = 1.0\n"
                "sustain_evals = 2\n"
                "[[train.performance_gate.stages]]\n"
                'fragment_resets = "false"\n'
            ),
        }
        for name, text in cases.items():
            path = _write(tmp_path, text, f"{name}.toml")
            with pytest.raises(ValueError, match="bare TOML boolean"):
                load_run_config(path)

    def test_phase_labels_bad_key_and_collision_fail_loudly(self, tmp_path):
        # int() alone would accept "1_0" (-> 10), "+2", " 3 " -- silently
        # attaching the label to a different phase id.
        for index, bad in enumerate(
            ('"await"', '"1_0"', '"+2"', '" 3"', '"1.0"', '"١"')
        ):
            path = _write(
                tmp_path,
                f"[train.phase_labels]\n{bad} = \"serve\"\n",
                f"bad_{index}.toml",
            )
            with pytest.raises(ValueError, match="integer phase id"):
                load_run_config(path)
        path = _write(
            tmp_path,
            '[train.phase_labels]\n"1" = "a"\n"01" = "b"\n',
            "collide.toml",
        )
        with pytest.raises(ValueError, match=r"collide at phase id 1"):
            load_run_config(path)

    def test_phase_label_values_are_exempt_from_the_none_sentinel(
        self, tmp_path
    ):
        # A label is display text; phases legitimately named "none" or
        # "false" must stay strings, unlike every other table value.
        path = _write(
            tmp_path, '[train.phase_labels]\n0 = "none"\n1 = "false"\n'
        )
        assert load_run_config(path).train["phase_labels"] == {
            0: "none",
            1: "false",
        }

    def test_phase_labels_none_sentinel_disables(self, tmp_path):
        # phase_labels is Optional in TrainConfig, so wholesale disable
        # via the sentinel is supported (unlike model_kwargs below).
        # Any-case, like every other sentinel site.
        path = _write(tmp_path, '[train]\nphase_labels = "NONE"\n')
        config = load_run_config(path)
        assert "phase_labels" in config.train
        assert config.train["phase_labels"] is None

    def test_phase_labels_scalar_and_bad_values_fail_loudly(self, tmp_path):
        path = _write(tmp_path, "[train]\nphase_labels = 5\n")
        with pytest.raises(ValueError, match="must be a table"):
            load_run_config(path)
        # A mistyped dotted key ([train.phase_labels."0"] foo = "bar")
        # makes the value a sub-table; it must not reach TensorBoard tag
        # interpolation as a dict repr.
        path = _write(
            tmp_path,
            '[train.phase_labels."0"]\nfoo = "bar"\n',
            "subtable.toml",
        )
        with pytest.raises(ValueError, match="must be a string label"):
            load_run_config(path)

    def test_non_optional_train_keys_reject_the_none_sentinel(
        self, tmp_path
    ):
        # A None smuggled into a non-Optional TrainConfig field either
        # crashes mid-train() with the file unnamed (n_envs,
        # model_kwargs) or -- worse -- silently disables a falsy-checked
        # feature while config.json records null (record_video,
        # info_dict_eval, normalize_obs).
        for key in (
            "model_kwargs",
            "info_eval_survival_thresholds",
            "n_envs",
            "record_video",
            "info_dict_eval",
            "normalize_obs",
            "total_timesteps",
        ):
            path = _write(
                tmp_path, f'[train]\n{key} = "none"\n', f"{key}.toml"
            )
            with pytest.raises(ValueError, match=f"{key}.*not supported"):
                load_run_config(path)

    def test_optional_train_keys_accept_the_none_sentinel(self, tmp_path):
        # Optional fields are the sentinel's legitimate use.
        path = _write(
            tmp_path,
            "[train]\n"
            'early_stop_patience = "none"\n'
            'headline_key = "none"\n',
        )
        config = load_run_config(path)
        assert config.train["early_stop_patience"] is None
        assert config.train["headline_key"] is None

    def test_nested_survival_threshold_none_is_rejected(self, tmp_path):
        # Deep-merge cannot remove a recipe threshold via None; it would
        # only fail inside train() after everything spun up.
        path = _write(
            tmp_path,
            "[train.info_eval_survival_thresholds]\n"
            'bounce_count = "none"\n',
        )
        with pytest.raises(ValueError, match="bounce_count.*not supported"):
            load_run_config(path)

    def test_nested_tables_get_sentinels_and_boolean_rejection(
        self, tmp_path
    ):
        # One level down is where SB3's real knobs live
        # (policy_kwargs.ortho_init): conversion and rejection recurse.
        path = _write(
            tmp_path,
            "[train.model_kwargs.policy_kwargs]\n"
            'activation_fn = "none"\n',
        )
        config = load_run_config(path)
        assert config.train["model_kwargs"]["policy_kwargs"] == {
            "activation_fn": None
        }
        # raw keeps the file verbatim even below the top level: the
        # recursive conversion must build fresh containers, not mutate
        # the nested dicts train shares with raw.
        assert config.raw["train"]["model_kwargs"]["policy_kwargs"] == {
            "activation_fn": "none"
        }

    def test_scalar_array_elements_are_exempt_from_the_sentinel(
        self, tmp_path
    ):
        # A tuple-valued env kwarg has no meaningful None element; only
        # tables nested in arrays (gate stages) recurse.
        path = _write(tmp_path, '[env]\nphases = ["none", "a"]\n')
        assert load_run_config(path).env == {"phases": ["none", "a"]}
        for name, text in (
            (
                "nested_train",
                '[train.model_kwargs.policy_kwargs]\northo_init = "false"\n',
            ),
            ("nested_env", '[env.reward_weights]\nenable_bonus = "false"\n'),
        ):
            path = _write(tmp_path, text, f"{name}.toml")
            with pytest.raises(ValueError, match="bare TOML boolean"):
                load_run_config(path)

    def test_incomplete_performance_gate_fails_at_load(self, tmp_path):
        # Wholesale replacement means a partial gate is a broken gate;
        # it must fail here, not as a KeyError mid-train().
        path = _write(
            tmp_path,
            '[train.performance_gate]\nmetric_key = "bounce_count"\n',
        )
        with pytest.raises(ValueError, match=r"must define.*threshold"):
            load_run_config(path)
        path = _write(
            tmp_path,
            "[train.performance_gate]\n"
            'metric_key = "m"\n'
            "threshold = 1.0\n"
            "sustain_evals = 2\n"
            "stages = []\n",
            "empty_stages.toml",
        )
        with pytest.raises(ValueError, match="non-empty"):
            load_run_config(path)
        path = _write(
            tmp_path, "[train]\nperformance_gate = 5\n", "not_table.toml"
        )
        with pytest.raises(ValueError, match="must be a table"):
            load_run_config(path)

    def test_performance_gate_none_sentinel_disables(self, tmp_path):
        path = _write(tmp_path, '[train]\nperformance_gate = "none"\n')
        assert load_run_config(path).train["performance_gate"] is None

    def test_gate_unknown_keys_and_bad_scalar_types_fail_at_load(
        self, tmp_path
    ):
        # train() reads exactly four gate keys; anything else would be a
        # silent no-op recorded in config.json as applied.
        base = (
            "[train.performance_gate]\n"
            'metric_key = "bounce_count_ep_mean"\n'
            "threshold = 1.0\n"
            "sustain_evals = 2\n"
            "[[train.performance_gate.stages]]\n"
            "serve_vy_max = 1.2\n"
        )
        # The unknown key must sit in the gate table itself, before the
        # [[stages]] header, or TOML files it under the stage.
        path = _write(
            tmp_path,
            base.replace(
                "sustain_evals = 2\n",
                "sustain_evals = 2\nmetric_window = 5\n",
            ),
        )
        with pytest.raises(ValueError, match="metric_window"):
            load_run_config(path)
        # A near-miss typo also gets a did-you-mean hint.
        path = _write(
            tmp_path,
            base.replace(
                "threshold = 1.0\n", "thresold = 1.0\nthreshold = 1.0\n"
            ),
            "typo.toml",
        )
        with pytest.raises(
            ValueError, match="did you mean.*threshold"
        ):
            load_run_config(path)
        for name, text, message in (
            (
                "threshold",
                base.replace("threshold = 1.0", 'threshold = "none"'),
                "threshold must be a number",
            ),
            (
                "threshold_bool",
                base.replace("threshold = 1.0", "threshold = true"),
                "threshold must be a number",
            ),
            (
                "sustain",
                base.replace("sustain_evals = 2", "sustain_evals = 2.5"),
                "sustain_evals must be an integer",
            ),
            (
                "sustain_bool",
                base.replace("sustain_evals = 2", "sustain_evals = true"),
                "sustain_evals must be an integer",
            ),
            (
                "metric",
                base.replace(
                    'metric_key = "bounce_count_ep_mean"', "metric_key = 5"
                ),
                "metric_key must be a string",
            ),
        ):
            path = _write(tmp_path, text, f"{name}.toml")
            with pytest.raises(ValueError, match=message):
                load_run_config(path)

    def test_gate_stages_shape_is_fully_validated(self, tmp_path):
        head = (
            "[train.performance_gate]\n"
            'metric_key = "m"\n'
            "threshold = 1.0\n"
            "sustain_evals = 2\n"
        )
        for name, tail in (
            ("scalar", "stages = 5\n"),
            ("non_table_items", "stages = [1, 2]\n"),
            ("empty_table", "[[train.performance_gate.stages]]\n"),
        ):
            path = _write(tmp_path, head + tail, f"stages_{name}.toml")
            with pytest.raises(
                ValueError, match="non-empty array of non-empty tables"
            ):
                load_run_config(path)

    def test_gate_stage_values_get_the_none_sentinel(self, tmp_path):
        # Stage attrs are env kwargs; "none" must reach the env as None,
        # not as a truthy string, when the gate advances mid-run.
        path = _write(
            tmp_path,
            "[train.performance_gate]\n"
            'metric_key = "m"\n'
            "threshold = 1.0\n"
            "sustain_evals = 2\n"
            "[[train.performance_gate.stages]]\n"
            'early_touch_penalty = "none"\n',
        )
        gate = load_run_config(path).train["performance_gate"]
        assert gate["stages"] == [{"early_touch_penalty": None}]


class TestBuildTrainConfig:
    def test_file_overrides_recipe_and_deep_merges_model_kwargs(
        self, tmp_path
    ):
        path = _write(
            tmp_path,
            "[train]\n"
            "n_envs = 4\n"
            "[train.model_kwargs]\n"
            'ent_coef = "auto_0.05"\n',
        )
        cfg = build_train_config(
            "WallBallBootstrap", log_dir=str(tmp_path), config_file=path
        )
        assert cfg.n_envs == 4
        # The tweaked key wins; the recipe's calibrated bundle survives.
        recipe_kwargs = RECIPES["WallBallBootstrap"].extra_cfg["model_kwargs"]
        assert cfg.model_kwargs["ent_coef"] == "auto_0.05"
        assert cfg.model_kwargs["learning_starts"] == (
            recipe_kwargs["learning_starts"]
        )
        assert cfg.model_kwargs["buffer_size"] == recipe_kwargs["buffer_size"]
        # The recipe registry itself is never mutated.
        assert recipe_kwargs["ent_coef"] == "auto_0.02"
        assert cfg.run_config_file is not None
        assert cfg.run_config_file.path == str(path)

    def test_quick_test_and_explicit_kwargs_beat_the_file(self, tmp_path):
        path = _write(
            tmp_path,
            "[train]\ntotal_timesteps = 999_999\nn_envs = 4\n",
        )
        cfg = build_train_config(
            "WallBallBootstrap",
            log_dir=str(tmp_path),
            config_file=path,
            quick_test=True,
            n_envs=2,
        )
        # quick_test presets shrink whatever the file asked for, and the
        # explicit keyword wins over both.
        assert cfg.total_timesteps < 999_999
        assert cfg.n_envs == 2

    def test_env_table_reaches_both_envs_below_recipe_eval_overrides(
        self, tmp_path
    ):
        path = _write(
            tmp_path,
            "[env]\n"
            "paddle_joint_damping = 6.0\n"
            "serve_vy_max = 1.3\n"
            "[eval_env]\n"
            "episode_len = 900\n",
        )
        cfg = build_train_config(
            "WallBallBootstrap", log_dir=str(tmp_path), config_file=path
        )
        train_env = cfg.env_fn()
        try:
            # [env] reaches the training env.
            assert train_env.paddle_joint_damping == 6.0
            assert train_env.serve_vy_max == 1.3
        finally:
            train_env.close()
        eval_env = cfg.eval_env_fn()
        try:
            # Physics tweaks reach the eval env too (no silent split)...
            assert eval_env.paddle_joint_damping == 6.0
            # ...but the recipe's canonical eval serve still wins over
            # [env], and [eval_env] wins last.
            assert eval_env.serve_vy_max == 2.0
            assert eval_env.episode_len == 900
        finally:
            eval_env.close()

    def test_typoed_env_kwarg_fails_at_build_time(self, tmp_path):
        path = _write(tmp_path, "[env]\nserve_vy_maximum = 1.3\n")
        with pytest.raises((TypeError, ValueError)):
            build_train_config(
                "WallBallBootstrap", log_dir=str(tmp_path), config_file=path
            )

    def test_run_config_file_override_is_rejected(self, tmp_path):
        # Injecting run_config_file directly would record provenance for
        # a file the builder never loaded (or crash artifacts on a str).
        # Rejected ALWAYS: without a config_file...
        with pytest.raises(ValueError, match="config_file"):
            build_train_config(
                "WallBallBootstrap",
                log_dir=str(tmp_path),
                run_config_file="fake.toml",
            )
        # ...and with one (where it would clobber the loaded provenance).
        path = _write(tmp_path, "[train]\nn_envs = 4\n")
        with pytest.raises(ValueError, match="config_file"):
            build_train_config(
                "WallBallBootstrap",
                log_dir=str(tmp_path),
                config_file=path,
                run_config_file="fake.toml",
            )

    def test_explicit_env_factories_conflict_with_config_file(
        self, tmp_path
    ):
        # An explicit factory would silently discard the file's
        # [env]/[eval_env] tables while config.json claims they applied.
        path = _write(tmp_path, "[env]\nserve_vy_max = 1.3\n")
        for key in ("env_fn", "eval_env_fn"):
            with pytest.raises(ValueError, match=key):
                build_train_config(
                    "WallBallBootstrap",
                    log_dir=str(tmp_path),
                    config_file=path,
                    **{key: lambda: None},
                )

    def test_explicit_env_factories_stay_allowed_without_config_file(
        self, tmp_path
    ):
        # The conflict guard must not break the legitimate no-file API.
        def sentinel_factory():  # pragma: no cover - never called
            raise AssertionError

        cfg = build_train_config(
            "WallBallBootstrap",
            log_dir=str(tmp_path),
            env_fn=sentinel_factory,
        )
        assert cfg.env_fn is sentinel_factory

    def test_performance_gate_replaces_wholesale(self, tmp_path):
        path = _write(
            tmp_path,
            "[train.performance_gate]\n"
            'metric_key = "bounce_count_ep_mean"\n'
            "threshold = 1.0\n"
            "sustain_evals = 3\n"
            "[[train.performance_gate.stages]]\n"
            "serve_vy_max = 1.2\n"
            "[[train.performance_gate.stages]]\n"
            "serve_vy_max = 2.0\n",
        )
        cfg = build_train_config(
            "WallBallBootstrap", log_dir=str(tmp_path), config_file=path
        )
        gate = cfg.performance_gate
        assert gate["sustain_evals"] == 3
        # Wholesale replacement: the recipe's 4-stage ladder is gone.
        assert [stage["serve_vy_max"] for stage in gate["stages"]] == [
            1.2,
            2.0,
        ]


class TestArtifacts:
    def test_run_dir_gets_copy_and_provenance_block(self, tmp_path):
        from courtside_dynamics.training.artifacts import write_run_config

        text = "[train]\nn_envs = 4\n"
        path = _write(tmp_path, text)
        cfg = build_train_config(
            "WallBallBootstrap", log_dir=str(tmp_path), config_file=path
        )
        out = write_run_config(cfg, str(tmp_path))

        copied = tmp_path / "run_config.toml"
        assert copied.read_text(encoding="utf-8") == text

        payload = json.loads(open(out).read())
        block = payload["run_config_file"]
        assert block["path"] == str(path)
        assert block["sha256"] == cfg.run_config_file.sha256
        assert block["content"] == {"train": {"n_envs": 4}}

    def test_no_file_records_null_and_copies_nothing(self, tmp_path):
        from courtside_dynamics.training.artifacts import write_run_config

        cfg = build_train_config("WallBallBootstrap", log_dir=str(tmp_path))
        out = write_run_config(cfg, str(tmp_path))
        assert json.loads(open(out).read())["run_config_file"] is None
        assert not (tmp_path / "run_config.toml").exists()


def _starter_params():
    from courtside_dynamics.run_config import available_run_configs

    return [
        pytest.param(path, name, id=path.stem)
        for name, path in sorted(available_run_configs().items())
    ]


class TestStarterConfigs:
    """The checked-in configs/ starters stay loadable and drift-free.

    configs/README.md promises that running with a starter file is
    equivalent to running without it -- the file documents the tuning
    surface, it does not fork the calibration. When a recipe changes,
    the matching starter must be updated in the same commit.
    """

    _SKIP_FIELDS = {"env_fn", "eval_env_fn", "run_config_file"}

    def test_every_recipe_has_a_starter(self):
        from courtside_dynamics.run_config import available_run_configs

        catalog = available_run_configs()
        assert set(catalog) == set(RECIPES)
        # And no orphan TOMLs ship without a recipe to consume them.
        package_dir = next(iter(catalog.values())).parent
        stems = {path.stem for path in package_dir.glob("*.toml")}
        orphans = stems - {r.name_prefix for r in RECIPES.values()}
        assert not orphans, f"starters without a recipe: {orphans}"

    @pytest.mark.parametrize("toml_path, recipe_name", _starter_params())
    def test_starter_matches_its_recipe(
        self, toml_path, recipe_name, tmp_path
    ):
        file_cfg = build_train_config(
            recipe_name, log_dir=str(tmp_path), config_file=toml_path
        )
        plain_cfg = build_train_config(recipe_name, log_dir=str(tmp_path))
        for field in dataclasses.fields(type(plain_cfg)):
            if field.name in self._SKIP_FIELDS:
                continue
            assert getattr(file_cfg, field.name) == getattr(
                plain_cfg, field.name
            ), f"{toml_path.name}: TrainConfig.{field.name} drifted"

        env_keys = load_run_config(toml_path).env
        if not env_keys:
            return

        def norm(value):
            if isinstance(value, (list, tuple)):
                return tuple(value)
            return value

        env_file, env_plain = file_cfg.env_fn(), plain_cfg.env_fn()
        try:
            for key in env_keys:
                assert norm(getattr(env_file, key)) == norm(
                    getattr(env_plain, key)
                ), f"{toml_path.name}: env.{key} drifted"
        finally:
            env_file.close()
            env_plain.close()


class TestStarterHelpers:
    def test_copy_starter_config_copies_and_refuses_clobber(self, tmp_path):
        from courtside_dynamics.run_config import (
            available_run_configs,
            copy_starter_config,
        )

        dest = copy_starter_config("WallBallBootstrap", tmp_path / "cfgs")
        assert dest == tmp_path / "cfgs" / "wall_ball_bootstrap.toml"
        assert (
            dest.read_bytes()
            == available_run_configs()["WallBallBootstrap"].read_bytes()
        )
        # The copy is loadable and buildable as-is.
        cfg = build_train_config(
            "WallBallBootstrap", log_dir=str(tmp_path), config_file=dest
        )
        assert cfg.run_config_file is not None

        # User edits must never be silently reset...
        dest.write_text("[train]\nn_envs = 2\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="overwrite=True"):
            copy_starter_config("WallBallBootstrap", tmp_path / "cfgs")
        assert dest.read_text(encoding="utf-8") == "[train]\nn_envs = 2\n"
        # ...unless explicitly requested.
        copy_starter_config(
            "WallBallBootstrap", tmp_path / "cfgs", overwrite=True
        )
        assert b"n_envs = 8" in dest.read_bytes()

    def test_copy_starter_config_unknown_recipe(self, tmp_path):
        from courtside_dynamics.run_config import copy_starter_config

        with pytest.raises(KeyError, match="WallBallBootstrap"):
            copy_starter_config("NoSuchEnv", tmp_path)
