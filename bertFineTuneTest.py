from transformers import BertModel, BertTokenizer
import torch.nn as nn

# 모델 정의
class ColorPredictor(nn.Module):
    def __init__(self):
        super(ColorPredictor, self).__init__()
        self.bert = BertModel.from_pretrained("C:/Users/MIN-VCI/Desktop/w2vTest/translate/wrtnModel20")
        self.regressor = nn.Linear(self.bert.config.hidden_size, 3)  # HSV는 3차원 값을 가짐

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hsv_values = self.regressor(outputs[1])
        return hsv_values


# 모델 사용
model = ColorPredictor()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 질문 설정
question = "plum top and red-violet bottom"

# 질문을 모델이 이해할 수 있는 형식으로 변환
inputs = tokenizer(question, return_tensors="pt")

# 변환된 질문을 모델에 입력하여 HSV 값을 예측
outputs = model(**inputs)

print(outputs)