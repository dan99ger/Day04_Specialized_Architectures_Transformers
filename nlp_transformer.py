import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_transformer_pipeline():
    print("--- 1. تحميل Tokenizer والنموذج المسبق DistilBERT ---")
    model_name = "distilbert-base-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # 2. نص تجريبي لمعالجته
    texts = [
        "This Machine Learning course is absolutely amazing and helpful!",
        "The code was buggy and the explanation was terrible."
    ]
    labels = torch.tensor([1, 0]).to(device)  # 1: إيجابي, 0: سلبي

    # 3. تحويل النصوص إلى Tokens و Input IDs
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)

    print("\n[Tokenized Output Keys]:", inputs.keys())
    print("Shape of input_ids:", inputs['input_ids'].shape)

    # 4. التمرير الأمامي وحساب الخسارة
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits

    print(f"\nحسب دالة الخسارة (Loss): {loss.item():.4f}")
    
    # 5. تحويل الـ Logits إلى التوقعات النهائية
    predictions = torch.argmax(logits, dim=-1)
    print("التوقعات (Predictions):", predictions.cpu().numpy())

    # 6. حفظ النموذج والترميز (Tokenizer)
    os.makedirs('artifacts', exist_ok=True)
    model.save_pretrained('artifacts/fine_tuned_distilbert')
    tokenizer.save_pretrained('artifacts/fine_tuned_distilbert')
    print("\nتم حفظ نموذج Transformer والرمز في artifacts/fine_tuned_distilbert")

if __name__ == '__main__':
    run_transformer_pipeline()