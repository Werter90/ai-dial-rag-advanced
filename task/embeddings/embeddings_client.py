import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str):
        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self._api_key = api_key

    def get_embeddings(self, inputs: list[str], dimensions: int) -> dict[int, list[float]]:
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }
        request_data = {
            "input": inputs,
            "dimensions": dimensions,
        }
        response = requests.post(url=self._endpoint, headers=headers, json=request_data, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return {item["index"]: item["embedding"] for item in data["data"]}
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
