import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications import MobileNetV3Large
from keras.applications.mobilenet_v3 import preprocess_input
from keras.optimizers import Adam

# ベースモデルを立ち上げる
def build_base_model() -> tf.keras.Model:
    # ImageNet学習済みモデルを特徴抽出器として利用
    base_model = MobileNetV3Large(weights='imagenet', include_top=False)
    # 転移学習の初期段階では重みを固定
    base_model.trainable = False
    return base_model

# 転移学習モデル（凍結状態）を構築する
def create_mobilenet_model(data_augmentation,base_model,train_ds, val_ds, class_names):
    model = models.Sequential([ 
        layers.Input(shape=(None, None, 3)),
        data_augmentation,        # データ拡張
        layers.Resizing(256, 256, crop_to_aspect_ratio=True),  # アスペクト比を維持したままリサイズ
        layers.RandomCrop(224, 224),                           # 学習時にランダムクロップして汎化性能を向上
        base_model,                           
        layers.GlobalAveragePooling2D(),      # 特徴マップをベクトル化
        layers.Dense(512, activation='relu'), # 分類ヘッド
        layers.Dropout(0.3),                  
        layers.Dense(len(class_names), activation='softmax') # 出力層（4択の動物） 
    ])
    
    #学習データと検証データの正規化
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    # モデルのコンパイル
    model.compile(optimizer=Adam(learning_rate=0.0001),  
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return train_ds, val_ds, model

# 学習の打ち切りと学習率調整によって、過学習の抑制と学習の安定化を行う
def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    
    return [early_stopping, reduce_lr]
 
# ベースモデルの後半だけを学習可能にして微調整する
def fine_tune_model(model: tf.keras.Model, base_model: tf.keras.Model, fine_tune_at: int = 100):
    """
    ベースモデルの後半層を解放し、低学習率で再学習を行う（fine-tuning）
    """
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]: 
        layer.trainable = False 
        
    for layer in base_model.layers[fine_tune_at:]: 
        layer.trainable = True 

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    return model

# 転移学習とFine-tuningの学習履歴を結合
def merge_history(h1,h2):
    history = {}
    for key in h1.history.keys():
        history[key] = h1.history[key] + h2.history.get(key, [])
    return history
    
