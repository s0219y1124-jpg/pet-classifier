from keras.utils import image_dataset_from_directory
from keras.applications.mobilenet_v3 import preprocess_input
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import shutil
import uuid
from PIL import Image
import glob
import matplotlib.pyplot as plt

# テスト用データセットを作成する
def create_testdatas(
    data_dir,
    class_names,
    batch_size=32
):
    test_ds = image_dataset_from_directory(
        data_dir,            
        batch_size=batch_size,         
        class_names=class_names,  
        shuffle=False
    )
    
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))
    
    return test_ds

#混合行列を用いて4種のテストデータを分析する関数
def confusion_analysis(test_ds, model):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
    
        pred_labels = np.argmax(preds, axis=1)  
    
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
#誤認画像の保存を行う関数
def save_misclassified(save_root,test_ds,model):

    if save_root.exists():
        for item in save_root.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    os.makedirs(save_root, exist_ok=True)

    misclassified_images = []
    correct_conf = []
    incorrect_conf = []

    for images, labels in test_ds:
        preds = model.predict(images)
        pred_labels = np.argmax(preds, axis=1)

        for k in range(len(labels)):
            true = labels[k].numpy()
            pred = pred_labels[k]
            confidence = np.max(preds[k])
        
            #正解データと誤認データの確信度をそれぞれ格納
            if true == pred:
                correct_conf.append(confidence)
            else:
                incorrect_conf.append(confidence)
            
            # trueラベル→予測ラベルごとに保存
            if true != pred:
                
                # フォルダ名作成（例: 0_to_1）
                folder_name = f"{true}_to_{pred}"
                save_dir = os.path.join(save_root, folder_name)
                os.makedirs(save_dir, exist_ok=True)

                # 画像保存
                img = images[k].numpy().astype("uint8")
                img_pil = Image.fromarray(img)

                filename = f"{uuid.uuid4().hex}.jpg"
                img_pil.save(os.path.join(save_dir, filename))
            
                misclassified_images.append((img, true, pred, confidence))
    
    return misclassified_images,correct_conf,incorrect_conf

#誤認画像を見るための関数
def check_misclassified(misclassified_images, class_names,save_root):
    jedge = 'y'
    while jedge != 'n':
        jedge = input("誤認画像を確認する？(y/n)>> ")
        if jedge == 'y':
            print(os.listdir(save_root))
            target_folder = input("見たいフォルダ名を入力 (例: 3_to_1)>> ")
        
            if not os.path.exists(os.path.join(save_root, target_folder)):
                print("そのフォルダは存在しないよ")
                continue

            files = glob.glob(os.path.join(save_root, target_folder, "*.jpg"))
        
            print(f"{target_folder}: {len(files)}件")

            target_true, target_pred = map(int, target_folder.split("_to_"))

            filtered = [
                (img, true, pred, conf)
                for img, true, pred, conf in misclassified_images
                if true == target_true and pred == target_pred
            ]

            for img, true, pred, conf in filtered[:10]:
                plt.rcParams['font.family'] = 'Meiryo'
                plt.imshow(img)
                plt.title(
                    f"true: {class_names[true]} / "
                    f"pred: {class_names[pred]} / "
                    f"conf: {conf:.2f}"
                )
                plt.axis('off')
                plt.show()
                
#その他の画像を分析し、確信度を算出
def other_analysis(other_ds,model):
    other_conf = []

    for images, labels in other_ds:

        predictions = model.predict(images, verbose=0)

        # 各画像の最大confidence
        max_scores = np.max(predictions, axis=1)

        other_conf.extend(max_scores)
        
    return other_conf