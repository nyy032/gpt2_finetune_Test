import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

# 데이터 로딩
with open('color_dataset.json', 'r') as file:
    data = json.load(file)

# BERT 토크나이저 및 모델 불러오기
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 출력 레이블의 개수에 맞게 조정

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

total_size = len(data)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_data, val_data = random_split(data, [train_size, val_size])

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item['context'] + ' ' + item['question'],
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        # BERT 모델 입력 구성
        inputs = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'token_type_ids': inputs['token_type_ids'].squeeze(),
        }
        labels = torch.tensor([item['answer']['hue'], item['answer']['saturation'], item['answer']['value']], dtype=torch.float32)
 # 레이블을 Float로 변환
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)

        return inputs, labels

# 데이터셋 및 데이터로더 생성
train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(val_data, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 옵티마이저 및 손실 함수 설정
criterion = torch.nn.MSELoss()  # 평균 제곱 오차 사용
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 10

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_train_loss = total_loss / len(train_dataloader)

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_dataloader:
            val_inputs = {key: value.to(device) for key, value in val_inputs.items()}
            val_labels = val_labels.to(device)
            val_outputs = model(**val_inputs)
            val_loss = criterion(val_outputs.logits, val_labels)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_dataloader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

# 모델 저장
model.save_pretrained("tuned_model")

# 토크나이저도 함께 저장하려면
tokenizer.save_pretrained("tuned_tokenizer")
