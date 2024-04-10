import mlora
from mlora.model.modelargs import MultiLoraBatchData
from mlora.utils import setup_seed

train_config = {
    "cutoff_len": 256,
    "save_step": 2000,
    "early_stop_test_step": 2000,
    "train_lora_candidate_num": 2,
    "train_lora_simultaneously_num": 1,
    "train_strategy": "pipe"
}

class FakeDispatcher:
    def get_train_data() -> MultiLoraBatchData:
        pass

    def check_task_done() -> bool:
        pass

def train_worker(rank: int, world_size: int, args):
    setup_seed(42)
    _, model = mlora.load_base_model(args.base_model,
                                     "llama",
                                     args.device,
                                     args.load_4bit,
                                     args.load_8bit,
                                     None)
    mlora.init_lora_model(model, config.lora_configs_)
    dispatcher = FakeDispatcher()
    pipe = mlora.Pipe(model,
                        config,
                        dispatcher,
                        args.device,
                        args.rank,
                        args.balance)
    pass

if __name__ == "__main__":
    train_worker()