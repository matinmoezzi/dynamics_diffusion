from hydra.core.hydra_config import HydraConfig
import re
from omegaconf import DictConfig, OmegaConf, open_dict, read_write


# Custom resolver that acts like a simple if-else statement
def if_resolver(condition, true_val, false_val):
    return true_val if condition else false_val


def steps_to_human_readable(step_count) -> str:
    # Ensure the input is an integer
    step_count = int(step_count)

    # Convert to human-readable format
    if step_count < 1000:
        return str(step_count)
    elif step_count < 1000000:
        return f"{step_count/1000:.0f}K"  # for thousands
    else:
        return f"{step_count/1000000:.0f}M"  # for millions


def sde_continuous_solver(node):
    if hasattr(node, "continuous"):
        if node.continuous:
            return "cont"
    return ""


def get_runtime_choice(key):
    instance = HydraConfig.get()
    return instance.runtime.choices[f"{key}@trainer.{key}"]


def karras_distillation(training_mode):
    if training_mode == "progdist":
        return False
    elif "consistency" in training_mode:
        return True
    else:
        raise ValueError(f"Unknown training mode {training_mode}")


def del_key(cfg: DictConfig, key: str):
    with read_write(cfg):
        with open_dict(cfg):
            del cfg[key]


def extract_step_number(path: str):
    # Extract step number using regex from the path
    match = re.search(r"checkpoint_(\d+).pt", path)
    if match:
        return str(int(match.group(1)))
    return 0  # Return -1 or any default value if pattern not found
