from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Dict
import io
from pathlib import Path
import logging
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#フロントエンドのアクセスを許可する
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

logger=logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# モデルの存在チェック
model_path = BASE_DIR.parent / "models" / "pet_model.keras"
if not model_path.exists():
    raise RuntimeError("モデルが存在しません")

# モデルロード&チェック
try:
    model = load_model(model_path)
    logger.info("モデルの読み込みに成功しました")
except Exception as e:
    raise RuntimeError(f"モデル読み込みに失敗 {e}")

#クラス名の準備
CLASS_NAMES = ['うさぎ', 'ねずみ', '犬', '猫']

#使用モデル
MODEL_TYPE = "efficientnet"

#レスポンスの型を指定
class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    scores: Dict[str, float]
    inference_time: float

#予測関数の準備
def predict_image(img_array: np.ndarray):
    start = time.time()

    # 256x256へ
    img_array = tf.image.resize(img_array, (256, 256))

    # 中央224x224切り抜き
    img_array = tf.image.central_crop(img_array, 224/256)
    
    """
    モデルデータ用に正規化
    MobileNet系統を使う場合は正規化が必要だが
    EfficientNet系統を使う場合は不要
    """

    if MODEL_TYPE == "mobilenet":
        from keras.applications.mobilenet_v3 import preprocess_input
        USE_PREPROCESS_INPUT = True

    elif MODEL_TYPE == "efficientnet":
        USE_PREPROCESS_INPUT = False

    if USE_PREPROCESS_INPUT:
        img_array = preprocess_input(img_array)

    img_tensor = np.expand_dims(img_array, axis=0)

    #モデルを予測
    prediction = model.predict(img_tensor, verbose=0)
    predicted_index = np.argmax(prediction[0]) # 一番確率が高いインデックス 
    max_score = float(np.max(prediction[0])) #一番確率が高いインデックスのスコア
    scores = {
        CLASS_NAMES[i]: round(float(prediction[0][i]), 4)
        for i in range(len(CLASS_NAMES))
        }
    
    if max_score < 0.9:
        label = "判別不可"
    else:
        label = CLASS_NAMES[predicted_index]
    
    end=time.time()
    inference_time = round(end - start, 3)

    return label, scores, inference_time

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    logger.info(f"received file: {file.filename}")
    
    #ファイル形式チェック
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルを入れてください")
    
    #画像ファイルをチェック
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
    except Exception as e:
        logger.error(f"画像処理エラー: {e}")
        raise HTTPException(status_code=400, detail="画像ファイルが読み込めませんでした。")

    try:
        label, scores, inference_time=predict_image(img_array)
        logger.info(
            f"prediction result: file={file.filename}, "
            f"label={label}, scores={scores}, "
            f"time={inference_time:.3f}s"
            )
        
    except Exception as e:
        logger.error(f"予測エラー: {e}")
        raise HTTPException(status_code=500, detail="予測中にエラーが発生しました")
    
    return {
        "success": True,
        "prediction": label,
        "scores": scores,
        "inference_time": float(inference_time)
    }