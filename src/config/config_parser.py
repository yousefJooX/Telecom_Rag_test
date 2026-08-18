import os
import yaml
from dotenv import load_dotenv

# for  load  the  .env secrets
load_dotenv()

class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
            """
             src/config/, so 3× dirname climbs 3 levels:
            .../Telecom_Rag_test/src/config/config_parser.py
                  ^                ^
                  dirname×3 → (project root)
            
            
            """
            config_path = os.path.join(base_dir, "config", "config.yml")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
            
    @property
    def app_name(self) -> str:
        return self._config["app"]["name"]

    @property
    def app_version(self) -> str:
        return self._config["app"]["version"]

    @property
    def vector_index_path(self) -> str:
        return self._config["data"]["vector_index_path"]

    @property
    def chunk_size(self) -> int:
        return int(self._config["data"]["chunk_size"])

    @property
    def chunk_overlap(self) -> int:
        return int(self._config["data"]["chunk_overlap"])

    @property
    def batch_size(self) -> int:
        return int(self._config["data"].get("batch_size", 50))

    @property
    def embedding_provider(self) -> str:
        return self._config["models"]["embedding_provider"]

    @property
    def embedding_model_name(self) -> str:
        return self._config["models"]["embedding_model_name"]

    @property
    def llm_provider(self) -> str:
        return self._config["models"]["llm_provider"]

    @property
    def llm_model_name(self) -> str:
        return self._config["models"]["llm_model_name"]

    @property
    def temperature(self) -> float:
        return float(self._config["models"]["temperature"])

    @property
    def k_retrieval(self) -> int:
        return int(self._config["models"]["k_retrieval"])

    @property
    def google_api_key(self) -> str:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set in .env file!")
        return key

settings = Config()