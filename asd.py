import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 가상의 데이터셋 예시
color_data = {
    "red": (346, 96, 93),
    "maroon": (346, 83, 76),
    "scarlet": (350, 94, 99),
    "orange": (25, 80, 100),
    "green-yellow": (54, 44, 95),
    "olive green": (59, 49, 71),
    "green": (135, 65, 65),
    "forest green": (140, 43, 65),
    "aquamarine": (186, 36, 91),
    "sky blue": (190, 50, 92),
    "middle blue": (190, 45, 90),
    "blue-green": (191, 100, 72),
    "cerulean": (193, 99, 83),
    "green-blue": (204, 80, 78),
    "navy blue": (210, 100, 80),
    "denim": (213, 89, 74),
    "blue": (216, 100, 100),
    "bluetiful": (224, 74, 91),
    "violet-blue": (245, 45, 78),
    "blue-violet": (249, 53, 72),
    "black": (0, 0, 0),
    "white": (0, 0, 100),
    "gray": (0, 0, 50),
    "brown": (0, 75, 65),
    "orange": (34, 100, 100),
    "yellow": (60, 100, 100),
    "khaki": (58, 91, 43),
    "mint": (180, 100, 100),
    "skyblue": (203, 46, 98),
    "purple": (300, 100, 50),
    "pink": (330, 59, 100),
    "beige": (36, 79, 77),
    "pink flamingo": (300, 54, 99),
    "red-violet": (324, 73, 73),
    "orchid": (314, 31, 89),
    "plum": (314, 65, 56),
}

with open("output.txt", "w") as file:
    for key1, value1 in color_data.items():
        for key2, value2 in color_data.items():
            #output_line = '{{"input": "{} top and {} bottom", "output": "{}, {}"}},\n'.format(key1, key2, value1, value2)
            output_line = '{{"context": "{} is {} and {} is {}", "question": "{} top and {} bottom", "answer": "{}, {}" }},\n'.format(key1, value1, key2, value2, key1, key2, value1, value2)
            file.write(output_line)

print("파일이 생성되었습니다.")


#"cadet blue": (219, 13, 76),
    #"brick red": (352, 77, 78),
    #"english vermilion": (358, 65, 80),
    #"madder lake": (359, 75, 80),
    #"permanent geranium lake": (0, 80, 88),
    #"maximum red": (0, 85, 85),
    #"indian red": (3, 61, 73),
    #"orange-red": (3, 71, 100),
    #"sunset orange": (4, 75, 100),
    #"bittersweet": (6, 63, 100),
    #"dark venetian red": (10, 80, 70),
    #"venetian red": (10, 70, 80),
    #"light venetian red": (10, 60, 90),
    #"vivid tangerine": (12, 50, 100),
    #"middle red": (15, 50, 90),
    #"burnt orange": (18, 71, 100),
    #"red-orange": (20, 88, 100),
    #"macaroni and cheese": (28, 52, 100),
    #"middle yellow red": (30, 50, 93),
    #"mango tango": (30, 100, 91),
    #"yellow-orange": (34, 74, 100),
    #"maximum yellow red": (40, 70, 95),
    #"banana mania": (44, 29, 98),
    #"maize": (44, 70, 95),
    #"orange-yellow": (45, 58, 97),
    #"goldenrod": (45, 59, 99),
    #"dandelion": (46, 63, 100),
    #"middle yellow": (55, 100, 100),
    #"spring green": (59, 20, 93),
    #"maximum yellow": (60, 78, 98),
    #"canary": (60, 40, 100),
    #"lemon yellow": (60, 38, 100),
    #"maximum green yellow": (65, 65, 90),
    #"middle green yellow": (72, 50, 75),
    #"inchworm": (75, 92, 89),
    #"light chrome green": (75, 67, 90),
    #"yellow-green": (76, 46, 88),
    #"maximum green": (90, 65, 55),
    #"asparagus": (92, 43, 63),
    #"granny smith apple": (112, 34, 88),
    #"fern": (126, 46, 72),
    #"middle green": (130, 45, 55),
    #"medium chrome green": (137, 35, 65),
    #"sea green": (149, 34, 87),
    #"shamrock": (160, 75, 80),
    #"mountain meadow": (162, 85, 70),
    #"jungle green": (163, 76, 67),
    #"caribbean green": (165, 100, 80),
    #"tropical rain forest": (168, 100, 46),
    #"middle blue green": (170, 35, 85),
    #"pine green": (175, 99, 47),
    #"maximum blue green": (180, 75, 75),
    #"robin's egg blue": (180, 100, 80),
    #"teal blue": (180, 100, 50),
    #"light blue": (180, 34, 85),
    #"turquoise blue": (186, 53, 91),
    #"outer space": (189, 22, 23),
    #"pacific blue": (192, 100, 77),
    #"maximum blue": (195, 65, 80),
    #"blue (i)": (196, 80, 90),
    #"cerulean blue": (200, 75, 80),
    #"cornflower": (201, 37, 92),
    #"midnight blue": (210, 100, 55),
    #"periwinkle": (223, 15, 90),
    #"blue": (224, 70, 90),
    #"wild blue yonder": (225, 34, 72),
    #"indigo": (227, 60, 78),
    #"manatee": (231, 12, 63),
    #"cobalt blue": (236, 30, 78),
    #"celestial blue": (240, 45, 80),
    #"blue bell": (240, 25, 80),
    #"maximum blue purple": (240, 25, 90),
    #"deep blue": (240, 100, 35),
    #"ultramarine blue": (250, 80, 75),
    #"middle blue purple": (260, 40, 75),
    #"purple heart": (263, 77, 76),
    #"royal purple": (267, 61, 63),
    #"violet (ii)": (274, 45, 64),
    #"medium violet": (280, 60, 70),
    #"wisteria": (281, 27, 86),
    #"lavender (i)": (287, 30, 80),
    #"vivid violet": (289, 62, 56),
    #"maximum purple": (290, 60, 50),
    #"purple mountains' majesty": (291, 21, 87),
    #"fuchsia": (300, 56, 76),
    #"violet (i)": (306, 60, 45),
    #"brilliant rose": (311, 55, 90),
    #"medium rose": (315, 50, 85),
    #"thistle": (320, 25, 92),
    #"mulberry": (323, 60, 78),
    #"middle purple": (325, 40, 85),
    #"maximum red purple": (325, 65, 65),
    #"jazzberry jam": (328, 93, 65),
    #"green": (120, 100, 50), 
""" # 가상의 데이터셋 생성
color_names = list(color_data.keys())
hsv_values = list(color_data.values())

# 데이터 전처리: 색상 이름을 숫자로 매핑
color_to_index = {color: index for index, color in enumerate(color_names)}
indexed_colors = [(color_to_index[color], hsv) for color, hsv in color_data.items()]

# 데이터 분할
train_data, test_data = train_test_split(indexed_colors, test_size=0.2, random_state=42)

# PyTorch 데이터셋 클래스 정의
class ColorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index, hsv = self.data[idx]
        return torch.tensor(index, dtype=torch.float32), torch.tensor(hsv, dtype=torch.float32)

# 데이터로더 생성
train_dataset = ColorDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 간단한 선형 회귀 모델 정의
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# 가상의 데이터셋 생성
color_names = list(color_data.keys())
hsv_values = list(color_data.values())

# 데이터 전처리: 색상 이름을 숫자로 매핑
color_to_index = {color: index for index, color in enumerate(color_names)}
indexed_colors = [color_to_index[color] for color in color_names]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(indexed_colors, hsv_values, test_size=0.2, random_state=42)

# PyTorch 데이터셋 클래스 정의
class ColorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        color_index = self.features[idx]
        color_hsv = torch.tensor(hsv_values[int(color_index)], dtype=torch.float32)
        return color_hsv, self.labels[idx]

# 데이터로더 생성
train_dataset = ColorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 간단한 선형 회귀 모델 정의
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# 모델, 손실 함수, 최적화 함수 초기화
input_size = 3  # HSV 값
output_size = 3  # HSV 값
model = LinearRegressionModel(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 학습 진행
num_epochs = 1000
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 테스트 데이터로 모델 평가
model.eval()
with torch.no_grad():
    test_inputs = torch.tensor([data[1] for data in test_data], dtype=torch.float32)
    predictions = model(test_inputs)

print("테스트 데이터 예측 결과:")
for i, (index, actual_hsv) in enumerate(test_data):
    color_name = color_names[index]
    predicted_hsv = predictions[i].tolist()
    print(f"{color_name}: Actual HSV {actual_hsv}, Predicted HSV {predicted_hsv}")


import colorsys

# 사용자에게 입력 받기
user_description = input("예를 들어 white 상의에 black 바지를 입은 사람의 색상을 설명해주세요: ")

# 사용자 입력을 기반으로 모델에 전달할 데이터 생성
def color_description_to_hsv(description):
    # 여러 색상을 포함하는 설명을 받아 각 색상의 평균 HSV 값을 계산
    colors = description.split(" ")
    hsv_sum = [0, 0, 0]
    count = 0

    for color in colors:
        if color in color_names:
            color_index = color_to_index[color]
            hsv_values = hsv_data[color_index]
            hsv_sum = [sum(x) for x in zip(hsv_sum, hsv_values)]
            count += 1

    if count > 0:
        avg_hsv = [x / count for x in hsv_sum]
        return avg_hsv
    else:
        return None

user_color_hsv = color_description_to_hsv(user_description)

if user_color_hsv:
    # 모델에 입력값 전달하여 예측
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(user_color_hsv, dtype=torch.float32).view(1, -1))

    # 예측된 HSV 값을 출력
    predicted_hsv = prediction.squeeze().tolist()
    print(f"예측된 HSV 값: {predicted_hsv}")

    # 예측된 HSV 값을 RGB로 변환하여 시각적으로 확인
    predicted_rgb = colorsys.hsv_to_rgb(*[x / 100 for x in predicted_hsv])
    predicted_rgb = [int(x * 255) for x in predicted_rgb]
    print(f"예측된 RGB 값: {predicted_rgb}")

else:
    print("입력한 설명에 포함된 색상이 데이터셋에 존재하지 않습니다.") """