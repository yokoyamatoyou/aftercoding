from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Google Cloud Secret Managerクライアントをインポート
# pip install google-cloud-secret-manager
try:
    from google.cloud import secretmanager
except ImportError:  # pragma: no cover - optional dependency
    secretmanager = None

class AppSettings(BaseSettings):
    # .envファイルからの読み込みを有効にする
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # 環境変数 (本番環境ではGCPによって設定される)
    GCP_PROJECT_ID: Optional[str] = None
    ENVIRONMENT: str = "development"

    # .envファイルまたはSecret Managerから取得する値
    OPENAI_API_KEY: Optional[str] = None
    MAX_CONCURRENT_TASKS: int = 5

    def __init__(self, **values):
        super().__init__(**values)
        if self.ENVIRONMENT == "production":
            self._load_secrets_from_gcp()

    def _load_secrets_from_gcp(self):
        """本番環境の場合、GCP Secret Managerから機密情報を読み込む"""
        if secretmanager is None:
            raise RuntimeError(
                "google-cloud-secret-manager is required to load secrets from GCP"
            )
        if not self.GCP_PROJECT_ID:
            print("警告: 本番環境ですが、GCP_PROJECT_IDが設定されていません。")
            return

        client = secretmanager.SecretManagerServiceClient()

        # OpenAI APIキーの取得
        secret_name = f"projects/{self.GCP_PROJECT_ID}/secrets/OPENAI_API_KEY/versions/latest"
        try:
            response = client.access_secret_version(request={"name": secret_name})
            self.OPENAI_API_KEY = response.payload.data.decode("UTF-8")
            print("GCP Secret ManagerからOPENAI_API_KEYを正常に読み込みました。")
        except Exception as e:
            print(f"エラー: GCP Secret ManagerからのOPENAI_API_KEYの読み込みに失敗しました: {e}")

# アプリケーション全体でこのインスタンスを共有する
settings = AppSettings()
