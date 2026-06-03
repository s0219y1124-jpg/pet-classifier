from dataset import (
    convert_to_jpg,
    create_datasets,
    get_data_augmentation,
    count_dataset_size,
)
from model import (
    build_base_model,
    create_mobilenet_model,
    get_callbacks,
    fine_tune_model,
    merge_history,
)
from analysis import (
    create_testdatas,
    confusion_analysis,
    save_misclassified,
    check_misclassified,
    other_analysis,
)
from plot import (
    plot_histogram,
    plot_history
)
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import os
from pathlib import Path

def main():
    
    # 変換するフォルダのパス
    BASE_DIR = Path(__file__).resolve().parent
    input_folder = BASE_DIR.parent / "data" / "train_images" 
    check_folder = BASE_DIR.parent / "data" / "test_images" 
    extra_folder = BASE_DIR.parent / "data" / "other_images"
    
    #訓練データとテストデータの拡張子を統一化
    convert_to_jpg(input_folder)
    convert_to_jpg(check_folder)
    convert_to_jpg(extra_folder)
    
    # 訓練データと検証データのディレクトリを設定
    batch_size=32
    train_ds, val_ds = create_datasets(
        input_folder,
        img_size=(224, 224),
        batch_size=batch_size,
        seed=123,
        val_split=0.2
    )
    
    #訓練データのクラス名を表示
    print(train_ds.class_names)
    
    #データ拡張レイヤを作成
    data_augmentation=get_data_augmentation()
    
    #学習データの訓練用、検証用のデータ数をカウント
    count_dataset_size(train_ds, val_ds)
    
    #クラス名を保存
    class_names = train_ds.class_names
    
    #モデルの構築
    base_model = build_base_model()
    train_ds, val_ds, model=create_mobilenet_model(data_augmentation,base_model,train_ds, val_ds, class_names)
    early_stopping, reduce_lr=get_callbacks()
    
    # モデルの訓練
    h1 = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[early_stopping, reduce_lr]
    )
    
    base_model.trainable = True
    
    #fine_tune_atによる再学習を行う
    fine_tune_at = int(len(base_model.layers) * 0.8)
    model=fine_tune_model(model, base_model, fine_tune_at)
    
    h2 = model.fit( 
                train_ds, 
                epochs=10,
                validation_data=val_ds, 
                callbacks=[early_stopping, reduce_lr] ) 
    
    #1回目の学習と再学習のデータを結合する
    history=merge_history(h1,h2)
    
    # テストデータのディレクトリを設定
    test_ds = create_testdatas(
        check_folder,            # データのディレクトリ 
        class_names,             # クラス名を一致
        batch_size=32,            # バッチサイズ
    )
    
    # 学習結果の評価
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'検証データに対する精度: {test_acc:.2f}')
    print(f'検証データに対する損失: {test_loss:.2f}')
    
    #テストデータを混合行列で分析
    confusion_analysis(test_ds, model)
    
    #誤認画像を保存
    save_root = BASE_DIR.parent / "data" / "misclassified"
    misclassified_images,correct_conf,incorrect_conf=save_misclassified(save_root,test_ds,model)
    
    #誤認画像を確認
    check_misclassified(misclassified_images, class_names,save_root)
    
    """
    その他の画像を分析し、閾値をいくつにするか検証
    """
    
    # その他データのディレクトリを設定
    other_ds = image_dataset_from_directory(
        extra_folder,
        batch_size=32,
        shuffle=False
    )
    
    #テストデータを混合行列で分析
    other_conf=other_analysis(other_ds,model)
    
    """
    グラフの表示と保存
    """
    
    #フォントの種類を指定
    plt.rcParams['font.family'] = 'Meiryo'
    
    #保存フォルダの指定
    save_dir=BASE_DIR.parent / "learning_results" 
    os.makedirs(save_dir, exist_ok=True)
    
    # 正解データの確信度をグラフ化
    plot_histogram(correct_conf, "Confidence Distribution (Correct)", 'blue', "correct_data.png", save_dir)
    # 誤認データの確信度をグラフ化
    plot_histogram(incorrect_conf, "Confidence Distribution (InCorrect)", 'red', "incorrect_data.png", save_dir)
    # その他データの確信度をグラフ化（閾値検証用）
    plot_histogram(other_conf, "Confidence Distribution (Other)", 'green', "other_data.png", save_dir)
    
    #学習曲線の可視化
    plot_history(h1, history, save_dir)
    
    #学習済みモデルを保存
    model_save_path = BASE_DIR.parent / "models"
    os.makedirs(model_save_path, exist_ok=True)

    print("モデルを保存します...")
    model.save(model_save_path / "pet_model.keras")
    print("保存完了")
    
if __name__ == "__main__":
    main()  