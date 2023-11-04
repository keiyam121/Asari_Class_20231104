# 必要なライブラリをインポート
from fastapi import FastAPI

# FastAPI のインスタンス化
app = FastAPI()


# ルートディレクトリへの GET で Hellp World の表示
@app.get("/")
def root():
    return {"message": "Hello World"}