import requests

from logger.custom_logger import dp_logger


class Sender:
    def __init__(self, server_url, headers=None, auth=None, timeout=10):
        """サーバーへのデータ送信クラスの初期化"""
        self.server_url = server_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.auth = auth
        self.timeout = timeout
        dp_logger.info("ResultSenderの初期化が完了しました。")

    def send(self, data):
        try:
            response = requests.post(
                self.server_url,
                json=data,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout,  # タイムアウトを適宜設定
            )
            response.raise_for_status()  # HTTPエラーをチェック
            dp_logger.info(
                f"データの送信に成功しました。ステータスコード: {response.status_code}"
            )
            return True
        except requests.exceptions.RequestException as e:
            dp_logger.exception(f"データの送信中にエラーが発生しました: {e}")
            return False
