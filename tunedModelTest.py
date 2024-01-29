from transformers import BertTokenizerFast, AdamW, BertModel
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import json

class ColorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item['question'], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        hsv_values = [list(map(int, hsv.split(','))) for hsv in item['answer'][1:-1].split('), (')]
        inputs['labels'] = torch.tensor(hsv_values).float().view(-1)  # [H1, S1, V1, H2, S2, V2]
        return inputs

class ColorPredictor(nn.Module):
    def __init__(self):
        super(ColorPredictor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Linear(self.bert.config.hidden_size, 6)  # 2 HSV values

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hsv_values = self.regressor(outputs[1])
        return hsv_values

# 데이터 로드
with open('output.json', 'r') as f:
    data = json.load(f)

# 데이터셋과 데이터 로더 준비
dataset = ColorDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델, 손실 함수, 최적화 알고리즘 준비
model = ColorPredictor()
loss_fn = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 훈련 반복
for epoch in range(30):  # 전체 데이터를 10번 반복
    total_loss = 0
    for batch in dataloader:
        # 배치 데이터를 GPU로 이동
        input_ids = batch["input_ids"].squeeze(1).to(device)
        attention_mask = batch["attention_mask"].squeeze(1).to(device)
        labels = batch["labels"].to(device)
        
        # 모델 예측
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 손실 계산
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        
        # 역전파와 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader)}")
    
torch.save(model.state_dict(), 'model.pth')
print("Training complete.")