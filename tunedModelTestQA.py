""" import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel

class ColorPredictor(nn.Module):
    def __init__(self):
        super(ColorPredictor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Linear(self.bert.config.hidden_size, 6)  # 2 HSV values

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hsv_values = self.regressor(outputs[1])
        return hsv_values

# 모델 불러오기
model = ColorPredictor()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 평가 모드로 설정

# 토크나이저 초기화
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# 질문 설정
question = "What is the HSV value of red?"
print('내가 물어본 질문 : ',question)

# 질문을 토크나이저로 인코딩
inputs = tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
inputs.pop('token_type_ids', None)

# 모델에 질문 입력하고 HSV 값 예측
with torch.no_grad():  # 그라디언트 계산 비활성화
    outputs = model(**inputs)
    hsv_values = outputs.view(-1, 3).tolist()  # HSV 값 리스트로 변환

# HSV 값 출력
for hsv in hsv_values:
    print(f"H: {hsv[0]}, S: {hsv[1]}, V: {hsv[2]}") """

import matplotlib.colors as mcolors
import webcolors

def name_to_hsv(color_name):
    rgb_color = webcolors.name_to_rgb(color_name) # 색상 이름을 RGB 값으로 변환
    normalized_rgb = [x/255.0 for x in rgb_color] # RGB 값의 범위를 [0, 1]로 정규화
    hsv_color = mcolors.rgb_to_hsv(normalized_rgb) # RGB 값을 HSV 값으로 변환
    return hsv_color

# 예제 사용
blue_hsv = name_to_hsv('blue')
print(blue_hsv)