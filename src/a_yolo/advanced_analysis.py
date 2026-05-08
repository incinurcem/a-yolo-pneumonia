import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import os
import argparse

# Kendi projendeki model ve dataset sınıflarını buraya import etmelisin
# from model import AYOLO 
# from dataset import RSNADataset, create_dataloader

def enable_dropout(model):
    """MC Dropout için sadece Dropout katmanlarını train moduna alır."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def calculate_iou(boxA, boxB):
    """İki sınır kutusu (bounding box) arasındaki Intersection over Union değerini hesaplar."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) hesaplar."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    total_samples = len(y_prob)
    
    ece = 0
    for i in range(len(prob_true)):
        bin_weight = bin_counts[i] / total_samples
        ece += bin_weight * np.abs(prob_true[i] - prob_pred[i])
    return ece, prob_true, prob_pred

def run_advanced_analysis(model, dataloader, device, output_dir, mc_passes=10, iou_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    
    all_y_true = []
    all_y_prob = []
    
    # Hata Analizi Sayaçları
    error_counts = {
        'True_Positive': 0,
        'Localization_Error': 0, # Kutu buldu ama IoU düşük
        'Background_Error': 0,   # Yanlışlıkla boş alanda kutu buldu (False Positive)
        'False_Negative': 0      # Zatürre var ama model bulamadı
    }
    
    uncertainties = []

    model.eval()
    
    print("🚀 Gelişmiş Analiz Başlıyor (Calibration, Error Analysis & MC Dropout)...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Analyzing"):
            images = images.to(device)
            
            # 1. STANDART ÇIKARIM (Kalibrasyon ve Hata Analizi için)
            outputs = model(images)
            # Çıktıların modelinden nasıl döndüğüne bağlı olarak burayı uyarla:
            # probs = outputs['scores'].cpu().numpy()
            # preds_boxes = outputs['boxes'].cpu().numpy()
            
            # Örnek dummy ayrıştırma (Kendi çıktı yapına göre güncelle):
            probs = torch.sigmoid(outputs[:, 0]).cpu().numpy() # Örnek probability
            preds_boxes = outputs[:, 1:5].cpu().numpy() # Örnek kutular
            true_boxes = targets['boxes'].numpy()
            labels = targets['labels'].numpy()
            
            for i in range(len(probs)):
                prob = probs[i]
                pred_box = preds_boxes[i]
                true_box = true_boxes[i]
                has_pneumonia = labels[i] > 0
                
                all_y_prob.append(prob)
                all_y_true.append(1 if has_pneumonia else 0)
                
                # --- HATA ANALİZİ KATEGORİZASYONU ---
                if prob > 0.45: # Projendeki threshold (0.45 kullanmışsın)
                    if has_pneumonia:
                        iou = calculate_iou(pred_box, true_box)
                        if iou >= iou_threshold:
                            error_counts['True_Positive'] += 1
                        else:
                            error_counts['Localization_Error'] += 1
                    else:
                        error_counts['Background_Error'] += 1
                else:
                    if has_pneumonia:
                        error_counts['False_Negative'] += 1
            
            # 2. MC DROPOUT (Belirsizlik / Uncertainty için)
            enable_dropout(model) # Dropout'ları aç
            mc_probs = []
            for _ in range(mc_passes):
                mc_out = model(images)
                # Kendi çıktı yapına göre güncelle:
                mc_prob = torch.sigmoid(mc_out[:, 0]).cpu().numpy() 
                mc_probs.append(mc_prob)
            
            mc_probs = np.array(mc_probs)
            variance = np.var(mc_probs, axis=0) # N geçişteki varyans belirsizliği temsil eder
            uncertainties.extend(variance)
            
            model.eval() # Diğer batch için standart eval moduna dön

    # --- KALİBRASYON (ECE) & RELIABILITY DIAGRAM ---
    ece, prob_true, prob_pred = calculate_ece(all_y_true, all_y_prob, n_bins=10)
    print(f"\n📈 Expected Calibration Error (ECE): {ece:.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Mükemmel Kalibrasyon')
    plt.plot(prob_pred, prob_true, marker='o', label=f'A-YOLO (ECE={ece:.4f})')
    plt.xlabel('Ortalama Tahmin Edilen Güven (Confidence)')
    plt.ylabel('Gerçekleşme Oranı (Accuracy)')
    plt.title('Reliability Diagram (Güvenilirlik Diyagramı)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
    plt.close()
    
    # --- BELİRSİZLİK (UNCERTAINTY) DAĞILIMI ---
    plt.figure(figsize=(8, 6))
    plt.hist(uncertainties, bins=50, color='purple', alpha=0.7)
    plt.xlabel('Epistemik Belirsizlik (Varyans)')
    plt.ylabel('Örnek Sayısı')
    plt.title('MC Dropout Belirsizlik Dağılımı')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'uncertainty_distribution.png'))
    plt.close()

    # --- HATA ANALİZİ RAPORU ---
    total_samples = sum(error_counts.values())
    print("\n══════════════════════════════════════════════════")
    print("        🚀 HATA ANALİZİ (ERROR TAXONOMY) RAPORU")
    print("══════════════════════════════════════════════════")
    for key, val in error_counts.items():
        percentage = (val / total_samples) * 100 if total_samples > 0 else 0
        print(f"  {key.replace('_', ' '):<20}: {val:<5} ({percentage:.1f}%)")
    print("══════════════════════════════════════════════════")
    
    # Raporu CSV olarak kaydet
    pd.DataFrame([error_counts]).to_csv(os.path.join(output_dir, 'error_taxonomy.csv'), index=False)
    print(f"🎉 Tüm analiz sonuçları ve grafikler şuraya kaydedildi: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mc_passes', type=int, default=15)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- KENDİ PROJENİN CLASS'LARINI BURADA ÇAĞIRMALISIN ---
    # model = AYOLO(num_classes=2).to(device)
    # model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # dataloader = create_dataloader(args.csv_path, args.img_dir, batch_size=args.batch_size, is_train=False)
    
    # Yorum satırlarını kendi veri yapına göre açtıktan sonra fonksiyonu çağır:
    # run_advanced_analysis(model, dataloader, device, args.output_dir, mc_passes=args.mc_passes)
    
    print("Lütfen script içindeki model ve dataloader tanımlamalarını kendi projenize göre (AYOLO, RSNADataset) güncelleyiniz.")