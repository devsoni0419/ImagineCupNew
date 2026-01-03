from azure.ai.ml.entities import Model
from utils.azure_utils import get_ml_client, load_config

def register_model(model_path):
    ml_client = get_ml_client()
    cfg = load_config()

    model = Model(
        path=model_path,
        name=cfg["model"]["name"],
        version=cfg["model"]["version"]
    )

    ml_client.models.create_or_update(model)
