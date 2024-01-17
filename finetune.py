import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader



# 데이터셋 파일 경로
file_path = 'output.json'

# 파일 읽기
with open(file_path, 'r') as file:
    data = json.load(file)

# BERT 토크나이저 및 모델 불러오기
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 출력 레이블의 개수에 맞게 조정

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
            item['context'],
            item['question'],
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
        labels = torch.tensor(eval(item['answer']), dtype=torch.float32)  # 레이블을 Float로 변환
        return inputs, labels

# 데이터셋 및 데이터로더 생성
dataset = CustomDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 옵티마이저 및 손실 함수 설정
criterion = torch.nn.MSELoss()  # 평균 제곱 오차 사용
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 10

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)  # logits에서 손실을 계산
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
