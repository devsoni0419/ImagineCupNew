from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment
)
from utils.azure_utils import get_ml_client, load_config

cfg = load_config()
ml_client = get_ml_client()

endpoint = ManagedOnlineEndpoint(
    name=cfg["endpoint"]["name"],
    auth_mode="key"
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint.name,
    model=cfg["model"]["name"],
    instance_type="Standard_DS2_v2",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(deployment).result()
