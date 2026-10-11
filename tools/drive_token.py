"""Однократно получить OAuth-токен пользователя для выгрузки финалов на Google Drive (scope drive.file).

    python tools/drive_token.py client_secret.json [--out /workspace/farm/secrets/google_oauth_token.json]

Печатает ссылку: откройте её в браузере, войдите в аккаунт, где должны лежать финалы, нажмите «Разрешить».
Редирект идёт на http://localhost:8085 — на поде нужен SSH-туннель `-L 8085:localhost:8085`
(или запустите скрипт на своём ПК и скопируйте токен на под). Подробно — docs/SHEETS_SETUP.md, шаг 6.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("client_secret")
    ap.add_argument("--out", default=str(Path(os.environ.get("FARM_HOME", "/workspace/farm")) / "secrets" / "google_oauth_token.json"))
    ap.add_argument("--port", type=int, default=8085)
    a = ap.parse_args()
    from google_auth_oauthlib.flow import InstalledAppFlow  # ставится вместе с gspread
    flow = InstalledAppFlow.from_client_secrets_file(a.client_secret, SCOPES)
    creds = flow.run_local_server(host="localhost", port=a.port, open_browser=False,
                                  access_type="offline", prompt="consent")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(creds.to_json(), "utf-8")
    os.chmod(out, 0o600)
    print("токен сохранён:", out)


if __name__ == "__main__":
    main()
