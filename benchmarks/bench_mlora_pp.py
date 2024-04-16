import torch
import mlora
import random
import os
import torch.multiprocessing as mp
import signal

from typing import Dict, List
from mlora.config import LoraConfig, MLoRAConfig, TrainerConfig
from mlora.model.modelargs import LoraBatchDataConfig, MultiLoraBatchData
from mlora.utils import setup_seed

trainer_config = {
    "cutoff_len": 256,
    "save_step": 2000,
    "early_stop_test_step": 2000,
    "train_lora_candidate_num": 10,
    "train_lora_simultaneously_num": 1,
    "train_strategy": "pipe"
}


lora_config = {
    "name": "lora",
    "r": 16,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": {
        "q_proj": True,
        "k_proj": True,
        "v_proj": True,
        "o_proj": True,
        "w1_proj": False,
        "w2_proj": False,
        "w3_proj": False
    },
    "batch_size": 8,
    "micro_batch_size": 8,
    # unused
    "test_batch_size": 0,
    "num_epochs": 0,
    "data": "",
    "test_data": "",
    "prompt": "",
    "group_by_length": "",
    "expand_side": "",
    "optim": "sgd",
    "momentum": 0.0,
    "lr": 0.0,
}

class FakeConfig:
    lora_configs_: List[LoraConfig] = []
    trainer_config_: TrainerConfig = None

    def __init__(self):
        self.trainer_config_ = TrainerConfig(trainer_config)
        for idx in range(10):
            cur_lora_config = lora_config.copy()
            cur_lora_config['name'] = f"lora_{idx}"
            self.lora_configs_.append(LoraConfig(cur_lora_config))


class FakeDispatcher:
    _adapter_backward_cnt_: Dict[str, int] = {}
    _adapter_forward_cnt_: Dict[str, int] = {}
    _adapter_accumulation_step_: Dict[str, int] = {}
    labels_: torch.Tensor = None
    step_: int = 0
    total_step_: int = 0
    seq_len: int = 1024
    micro_batch_size : int = 8

    def __init__(self, config: MLoRAConfig, label: torch.Tensor, step: int):
        self.lables_ = label
        for lora_config in config.lora_configs_:
            adapter_name = lora_config.adapter_name_
            accumulation_step = lora_config.batch_size_ / lora_config.micro_batch_size_
            self._adapter_forward_cnt_[adapter_name] = 0
            self._adapter_backward_cnt_[adapter_name] = 0
            self._adapter_accumulation_step_[adapter_name] = accumulation_step
            self.total_step_ += step * accumulation_step

    def update_backward_cnt(self, adapter_name: str):
        self._adapter_backward_cnt_[adapter_name] += 1
        if self._adapter_backward_cnt_[adapter_name] == self._adapter_accumulation_step_[adapter_name]:
            self._adapter_forward_cnt_[adapter_name] = 0
            self._adapter_backward_cnt_[adapter_name] = 0

    def update_forward_cnt(self, adapter_name: str):
        self._adapter_forward_cnt_[adapter_name] += 1

    def __check_adapter_available(self, adapter_name: str) -> bool:
        return self._adapter_forward_cnt_[adapter_name] < self._adapter_accumulation_step_[adapter_name]

    def get_train_data(self) -> MultiLoraBatchData:
        for adapter_name in self._adapter_backward_cnt_:
            if self.__check_adapter_available(adapter_name):
                self.update_forward_cnt(adapter_name)
                self.step_ += 1
                return self.get_input(adapter_name)
        return None

    def check_task_done(self) -> bool:
        return self.step_ >= self.total_step_

    def get_input(self, adapter_name: str) -> MultiLoraBatchData:
        batch_tokens = []
        additional_masks = []
        lora_batch_data_config: List[LoraBatchDataConfig] = []

        start_idx = 0
        end_idx = 0

        for _ in range(0, self.micro_batch_size):
            tokens = [random.randint(1, 10000) for _ in range(self.seq_len)]
            batch_tokens.append(tokens)
            additional_masks.append([False] * self.seq_len)
            end_idx += 1

        lora_batch_data_config.append(LoraBatchDataConfig(
            adapter_name_=adapter_name,
            batch_start_idx_=start_idx,
            batch_end_idx_=end_idx,
        ))

        start_idx = end_idx
        return MultiLoraBatchData(batch_tokens_=batch_tokens,
                                  additional_mask_=additional_masks,
                                  lora_batch_data_config_=lora_batch_data_config,
                                  inference_model_=False)


def setup_labels(batch_size: int, seq_len: int) -> torch.Tensor:
    batch_input_ids = []
    for _ in range(0, batch_size):
        batch_input_ids.append([random.randint(1, 10000)
                               for _ in range(seq_len)])
    return torch.tensor(batch_input_ids, dtype=torch.long)


def train_worker(rank: int, world_size: int):
    setup_seed(42)
    config = FakeConfig()
    device = torch.device(f"cuda:{rank}")
    base_model = "/host_data/Llama-2-7b-hf/"
    balance = [9,8,8,10]
    partial_model_to_device = [
            index + sum(balance[:rank])for index in range(0, balance[rank])]
    _, model = mlora.load_base_model(base_model,
                                     "llama",
                                     device,
                                     False,
                                     False,
                                     partial_model_to_device)
    mlora.init_lora_model(model, config.lora_configs_)

    labels = setup_labels(2, 1024)
    dispatcher = FakeDispatcher(config, labels, 100)
    pipe = mlora.Pipe(model,
                      config,
                      dispatcher,
                      device,
                      rank,
                      balance)
    exit(pipe.run())


def main(rank: int, world_size: int):
    train_worker(rank, world_size)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    ps = []
    for i in range(4):
        p = mp.Process(target=main, args=(i, 4))
        p.start()
        ps.append(p)

    def kill_processes():
        for p in ps:
            os.kill(p.pid, signal.SIGTERM)

    def signal_handler(sig, frame):
        print('Termination signal received, killing processes...')
        kill_processes()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for p in ps:
        p.join()