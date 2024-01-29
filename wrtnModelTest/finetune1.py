import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaConfig, RobertaPreTrainedModel, RobertaTokenizer, AdamW, GPT2LMHeadModel, GPT2Tokenizer, GPT2PreTrainedModel, GPT2Model, GPT2Config
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, AdamW
from torch.nn import MSELoss
from tqdm import tqdm  # tqdm 라이브러리 추가
import json
from torch.optim import SGD

# 회귀 문제를 위한 BERT 모델 정의
""" class BertForSequenceRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs """

""" class RobertaForSequenceRegression(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs """

class GPT2ForSequenceRegression(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.gpt2 = GPT2Model(config)
        self.dropout = nn.Dropout(0.1)  # 여기���서 원���는 드롭아웃 비율��� 설정하실 수 있습니다.
        self.regressor = nn.Linear(config.hidden_size, 3)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.gpt2(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]
        pooled_output = hidden_states[:, -1]
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)

        # 여기��� labels가 제공되었을 때 손실을 계산합니다.
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs[2:]
        else:
            outputs = (logits,) + outputs[2:]

        return outputs  
# 데이터셋 클래스 정의
class ColorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        """ self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') """
        """ self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') """
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        color_info = example['color_info']
        hsv_value = example['hsv_value']

        inputs = self.tokenizer.encode_plus(
            color_info,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'hsv_value': torch.tensor(hsv_value, dtype=torch.float32)
        }

# 데이터셋 로드
with open('color_dataset.json', 'r') as f:
    data = json.load(f)

dataset = ColorDataset(data)
kk = 1e-3
number = 20
b_size = 2

# 모델 및 옵티마이저 초기화
""" model = BertForSequenceRegression.from_pretrained('bert-base-uncased') """
""" model = RobertaForSequenceRegression.from_pretrained("roberta-base") """
config = GPT2Config.from_pretrained("gpt2")
config.num_labels = 6  # 레이블의 수를 실제 값으로 변경하세요.
model = GPT2ForSequenceRegression.from_pretrained("gpt2", config=config)
optimizer = SGD(model.regressor.parameters(), lr=kk)



# 학습 함수 정의
def train(model, optimizer, dataset, batch_size, num_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0

        # tqdm으로 감싸서 진행률 표시
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            hsv_value = batch['hsv_value']

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=hsv_value)
            loss = outputs[0]
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f'Loss: {avg_loss}')

train(model, optimizer, dataset, batch_size=b_size, num_epochs=number)

# 새로운 입력에 대한 예측
def predict(model, tokenizer, input_text):
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].squeeze()
    attention_mask = inputs['attention_mask'].squeeze()

    with torch.no_grad():
        outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    
    predicted_hsv = outputs[0].squeeze().tolist()
    return predicted_hsv

# 예측 실행
input_text = "black"
predicted_hsv = predict(model, dataset.tokenizer, input_text)
print('0, 0, 0')
print(f"lr = '{kk}', num_epochs = '{number}'" )
print(f"Predicted HSV for '{input_text}': {predicted_hsv}")

input_text = "red"
predicted_hsv = predict(model, dataset.tokenizer, input_text)
print('0, 100, 100')
print(f"lr = '{kk}', num_epochs = '{number}'" )
print(f"Predicted HSV for '{input_text}': {predicted_hsv}")

input_text = "white"
predicted_hsv = predict(model, dataset.tokenizer, input_text)
print('0, 0, 100')
print(f"lr = '{kk}', num_epochs = '{number}'" )
print(f"Predicted HSV for '{input_text}': {predicted_hsv}")

input_text = "blue"
predicted_hsv = predict(model, dataset.tokenizer, input_text)
print('216, 100, 100')
print(f"lr = '{kk}', num_epochs = '{number}'" )
print(f"Predicted HSV for '{input_text}': {predicted_hsv}")

input_text = "blue"
predicted_hsv = predict(model, dataset.tokenizer, input_text)
print('216, 100, 100')
print(f"lr = '{kk}', num_epochs = '{number}'" )
print(f"Predicted HSV for '{input_text}': {predicted_hsv}")

input_text = "red bottom"
predicted_hsv = predict(model, dataset.tokenizer, input_text)
print('216, 100, 100')
print(f"lr = '{kk}', num_epochs = '{number}'" )
print(f"Predicted HSV for '{input_text}': {predicted_hsv}")