from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import yaml

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_ml_client():
    cfg = load_config()
    return MLClient(
        DefaultAzureCredential(),
        cfg["azure"]["subscription_id"],
        cfg["azure"]["resource_group"],
        cfg["azure"]["workspace_name"]
    )
