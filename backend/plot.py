import matplotlib.pyplot as plt
import os
import numpy as np

# 推論確信度の分布を可視化する
def plot_histogram(data, title, color, filename, save_dir):

    plt.figure(figsize=(6, 5))
    bins = np.arange(0, 1.1, 0.1)
    plt.hist(data, bins=bins, color=color, edgecolor="black")

    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.savefig(os.path.join(save_dir, filename))
    plt.show()
    plt.close()

def plot_history(h1,history, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    
    # 訓練と検証の損失をプロット
    plt.figure()
    plt.plot(history['loss'], label='訓練損失')
    plt.plot(history['val_loss'], label='検証損失')
    
    # fine-tuning 開始位置（h1のepoch数）
    plt.axvline(x=len(h1.history['loss']), linestyle="--", label="fine-tune start")
    
    plt.legend()
    plt.title("loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

    # 訓練と検証の精度をプロット
    plt.figure()
    plt.plot(history['accuracy'], label='訓練精度')
    plt.plot(history['val_accuracy'], label='検証精度')
    
    # fine-tuning 開始位置（h1のepoch数）
    plt.axvline(x=len(h1.history['accuracy']), linestyle="--", label="fine-tune start")
    
    plt.legend()
    plt.title("accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_dir, "accuracy.png"))
    plt.close()

