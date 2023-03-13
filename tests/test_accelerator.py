# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from unittest import mock

import pytest
import torch
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_habana.accelerator import AcceleratorHPU
from lightning_habana.strategies import StrategyParallelHPU, StrategySingleHPU
from tests.helpers import ClassifDataModule, ClassificationModel


def test_availability():
    assert AcceleratorHPU.is_available()


def test_device_name():
    assert AcceleratorHPU.get_device_name() == "GAUDI"


def test_accelerator_selected():
    trainer = Trainer(accelerator=AcceleratorHPU())
    assert isinstance(trainer.accelerator, AcceleratorHPU)


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
def test_all_stages(tmpdir, hpus):
    """Tests all the model stages using BoringModel on HPU."""
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=AcceleratorHPU(),
        devices=hpus,
        precision="16-mixed",
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_optimization(tmpdir):
    seed_everything(42)

    dm = ClassifDataModule(length=1024)
    model = ClassificationModel()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, accelerator=AcceleratorHPU(), devices=1)

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # validate
    result = trainer.validate(datamodule=dm)
    assert dm.trainer is not None
    assert result[0]["val_acc"] > 0.7

    # test
    result = trainer.test(model, datamodule=dm)
    assert dm.trainer is not None
    test_result = result[0]["test_acc"]
    assert test_result > 0.6

    # test saved model
    model_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(model_path)

    model = ClassificationModel.load_from_checkpoint(model_path)

    trainer = Trainer(default_root_dir=tmpdir, accelerator=AcceleratorHPU(), devices=1)

    result = trainer.test(model, datamodule=dm)
    saved_result = result[0]["test_acc"]
    assert saved_result == test_result


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
def test_stages_correct(tmpdir):
    """Ensure all stages correctly are traced correctly by asserting the output for each stage."""

    class StageModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            loss = loss.get("loss")
            # tracing requires a loss value that depends on the model.
            # force it to be a value but ensure we use the loss.
            loss = (loss - loss) + torch.tensor(1)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)
            x = loss.get("x")
            x = (x - x) + torch.tensor(2)
            return {"x": x}

        def test_step(self, batch, batch_idx):
            loss = super().test_step(batch, batch_idx)
            y = loss.get("y")
            y = (y - y) + torch.tensor(3)
            return {"y": y}

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            output = super().predict_step(batch, batch_idx)
            return (output - output) + torch.tensor(4)

    class TestCallback(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["loss"].item() == 1

        def on_validation_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["x"].item() == 2

        def on_test_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["y"].item() == 3

        def on_predict_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert torch.all(outputs == 4).item()

    model = StageModel()
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, accelerator=AcceleratorHPU(), devices=1, callbacks=TestCallback()
    )
    trainer.fit(model)
    trainer.test(model)
    trainer.validate(model)
    trainer.predict(model)


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
def test_accelerator():
    trainer = Trainer(accelerator=AcceleratorHPU(), devices=1)
    assert isinstance(trainer.accelerator, AcceleratorHPU)
    assert trainer.num_devices == 1

    trainer = Trainer(accelerator=AcceleratorHPU())
    assert isinstance(trainer.accelerator, AcceleratorHPU)
    assert trainer.num_devices == 8

    trainer = Trainer(accelerator="auto", devices=8)
    assert isinstance(trainer.accelerator, AcceleratorHPU)
    assert trainer.num_devices == 8


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
def test_accelerator_with_single_device():
    trainer = Trainer(accelerator=AcceleratorHPU(), devices=1)

    assert isinstance(trainer.strategy, StrategySingleHPU)
    assert isinstance(trainer.accelerator, AcceleratorHPU)


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
def test_accelerator_with_multiple_devices():
    trainer = Trainer(accelerator=AcceleratorHPU(), devices=8)

    assert isinstance(trainer.strategy, StrategyParallelHPU)
    assert isinstance(trainer.accelerator, AcceleratorHPU)


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
def test_accelerator_auto_with_devices_hpu():
    trainer = Trainer(accelerator="auto", devices=8)

    assert isinstance(trainer.strategy, StrategyParallelHPU)


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
def test_strategy_choice_single_strategy():
    trainer = Trainer(strategy=StrategySingleHPU(device=torch.device("hpu")), accelerator=AcceleratorHPU(), devices=1)
    assert isinstance(trainer.strategy, StrategySingleHPU)

    trainer = Trainer(accelerator=AcceleratorHPU(), devices=1)
    assert isinstance(trainer.strategy, StrategySingleHPU)


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
def test_strategy_choice_parallel_strategy():
    trainer = Trainer(
        strategy=StrategyParallelHPU(parallel_devices=[torch.device("hpu")] * 8),
        accelerator=AcceleratorHPU(),
        devices=8,
    )
    assert isinstance(trainer.strategy, StrategyParallelHPU)

    trainer = Trainer(accelerator=AcceleratorHPU(), devices=8)
    assert isinstance(trainer.strategy, StrategyParallelHPU)


def test_devices_auto_choice_hpu():
    trainer = Trainer(accelerator="auto", devices="auto")
    assert trainer.num_devices == 8


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
@pytest.mark.parametrize("hpus", [1])
def test_inference_only(tmpdir, hpus):
    model = BoringModel()

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, accelerator=AcceleratorHPU(), devices=hpus)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


def test_hpu_auto_device_count():
    assert AcceleratorHPU.auto_device_count() == 8


def test_hpu_unsupported_device_type():
    with pytest.raises(MisconfigurationException, match="`devices` for `HPUAccelerator` must be int, string or None."):
        Trainer(accelerator=AcceleratorHPU(), devices=[1])


def test_strategy_params_with_hpu_parallel_strategy():
    bucket_cap_mb = 100
    gradient_as_bucket_view = True
    static_graph = True
    find_unused_parameters = True
    strategy = StrategyParallelHPU(
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        find_unused_parameters=find_unused_parameters,
    )
    assert strategy._ddp_kwargs["bucket_cap_mb"] == bucket_cap_mb
    assert strategy._ddp_kwargs["gradient_as_bucket_view"] == gradient_as_bucket_view
    assert strategy._ddp_kwargs["static_graph"] == static_graph
    assert strategy._ddp_kwargs["find_unused_parameters"] == find_unused_parameters


@pytest.mark.xfail(MisconfigurationException, reason="Device should be HPU, got cpu instead.")  # ToDo
def test_multi_optimizers_with_hpu(tmpdir):
    class MultiOptimizerModel(BoringModel):
        def configure_optimizers(self):
            opt_a = torch.optim.Adam(self.layer.parameters(), lr=0.001)
            opt_b = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            return opt_a, opt_b

        def training_step(self, batch, batch_idx):
            opt1, opt2 = self.optimizers()
            loss = self.loss(self.step(batch))
            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()
            loss = self.loss(self.step(batch))
            opt2.zero_grad()
            self.manual_backward(loss)
            opt2.step()

    model = MultiOptimizerModel()
    model.automatic_optimization = False
    model.val_dataloader = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=AcceleratorHPU(),
        devices=1,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)


def test_hpu_device_stats_monitor():
    hpu_stats = AcceleratorHPU().get_device_stats("hpu")
    fields = [
        "Limit",
        "InUse",
        "MaxInUse",
        "NumAllocs",
        "NumFrees",
        "ActiveAllocs",
        "MaxAllocSize",
        "TotalSystemAllocs",
        "TotalSystemFrees",
        "TotalActiveAllocs",
    ]
    for f in fields:
        assert any(f in h for h in hpu_stats)
