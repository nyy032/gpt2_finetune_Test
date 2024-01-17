import torch
import torch.nn as nn
import torch.optim as optim



def find_color_in_sentence(color_data, sentence):
    hsv_values = []
    for color_entry in color_data:
        if color_entry['name'].lower() in sentence.lower():
            hsv_values.append(color_entry['hsv'])
    return hsv_values if hsv_values else None

# 가상의 데이터셋 예시
color_data = {
    "red": (346, 96, 93),
    "maroon": (346, 83, 76),
    "scarlet": (350, 94, 99),
    "brick red": (352, 77, 78),
    "english vermilion": (358, 65, 80),
    "madder lake": (359, 75, 80),
    "permanent geranium lake": (0, 80, 88),
    "maximum red": (0, 85, 85),
    "indian red": (3, 61, 73),
    "orange-red": (3, 71, 100),
    "sunset orange": (4, 75, 100),
    "bittersweet": (6, 63, 100),
    "dark venetian red": (10, 80, 70),
    "venetian red": (10, 70, 80),
    "light venetian red": (10, 60, 90),
    "vivid tangerine": (12, 50, 100),
    "middle red": (15, 50, 90),
    "burnt orange": (18, 71, 100),
    "red-orange": (20, 88, 100),
    "orange": (25, 80, 100),
    "macaroni and cheese": (28, 52, 100),
    "middle yellow red": (30, 50, 93),
    "mango tango": (30, 100, 91),
    "yellow-orange": (34, 74, 100),
    "maximum yellow red": (40, 70, 95),
    "banana mania": (44, 29, 98),
    "maize": (44, 70, 95),
    "orange-yellow": (45, 58, 97),
    "goldenrod": (45, 59, 99),
    "dandelion": (46, 63, 100),
    "yellow": (52, 55, 98),
    "green-yellow": (54, 44, 95),
    "middle yellow": (55, 100, 100),
    "olive green": (59, 49, 71),
    "spring green": (59, 20, 93),
    "maximum yellow": (60, 78, 98),
    "canary": (60, 40, 100),
    "lemon yellow": (60, 38, 100),
    "maximum green yellow": (65, 65, 90),
    "middle green yellow": (72, 50, 75),
    "inchworm": (75, 92, 89),
    "light chrome green": (75, 67, 90),
    "yellow-green": (76, 46, 88),
    "maximum green": (90, 65, 55),
    "asparagus": (92, 43, 63),
    "granny smith apple": (112, 34, 88),
    "fern": (126, 46, 72),
    "middle green": (130, 45, 55),
    "green": (135, 65, 65),
    "medium chrome green": (137, 35, 65),
    "forest green": (140, 43, 65),
    "sea green": (149, 34, 87),
    "shamrock": (160, 75, 80),
    "mountain meadow": (162, 85, 70),
    "jungle green": (163, 76, 67),
    "caribbean green": (165, 100, 80),
    "tropical rain forest": (168, 100, 46),
    "middle blue green": (170, 35, 85),
    "pine green": (175, 99, 47),
    "maximum blue green": (180, 75, 75),
    "robin's egg blue": (180, 100, 80),
    "teal blue": (180, 100, 50),
    "light blue": (180, 34, 85),
    "aquamarine": (186, 36, 91),
    "turquoise blue": (186, 53, 91),
    "outer space": (189, 22, 23),
    "sky blue": (190, 50, 92),
    "middle blue": (190, 45, 90),
    "blue-green": (191, 100, 72),
    "pacific blue": (192, 100, 77),
    "cerulean": (193, 99, 83),
    "maximum blue": (195, 65, 80),
    "blue (i)": (196, 80, 90),
    "cerulean blue": (200, 75, 80),
    "cornflower": (201, 37, 92),
    "green-blue": (204, 80, 78),
    "midnight blue": (210, 100, 55),
    "navy blue": (210, 100, 80),
    "denim": (213, 89, 74),
    "blue": (216, 100, 100),
    "cadet blue": (219, 13, 76),
    "periwinkle": (223, 15, 90),
    "blue": (224, 70, 90),
    "bluetiful": (224, 74, 91),
    "wild blue yonder": (225, 34, 72),
    "indigo": (227, 60, 78),
    "manatee": (231, 12, 63),
    "cobalt blue": (236, 30, 78),
    "celestial blue": (240, 45, 80),
    "blue bell": (240, 25, 80),
    "maximum blue purple": (240, 25, 90),
    "deep blue": (240, 100, 35),
    "violet-blue": (245, 45, 78),
    "blue-violet": (249, 53, 72),
    "ultramarine blue": (250, 80, 75),
    "middle blue purple": (260, 40, 75),
    "purple heart": (263, 77, 76),
    "royal purple": (267, 61, 63),
    "violet (ii)": (274, 45, 64),
    "medium violet": (280, 60, 70),
    "wisteria": (281, 27, 86),
    "lavender (i)": (287, 30, 80),
    "vivid violet": (289, 62, 56),
    "maximum purple": (290, 60, 50),
    "purple mountains' majesty": (291, 21, 87),
    "fuchsia": (300, 56, 76),
    "pink flamingo": (300, 54, 99),
    "violet (i)": (306, 60, 45),
    "brilliant rose": (311, 55, 90),
    "orchid": (314, 31, 89),
    "plum": (314, 65, 56),
    "medium rose": (315, 50, 85),
    "thistle": (320, 25, 92),
    "mulberry": (323, 60, 78),
    "red-violet": (324, 73, 73),
    "middle purple": (325, 40, 85),
    "maximum red purple": (325, 65, 65),
    "jazzberry jam": (328, 93, 65),
    "black": (0, 0, 0),
    "white": (0, 0, 100),
    "gray": (0, 0, 50),
    "brown": (0, 75, 65),
    "orange": (34, 100, 100),
    "yellow": (60, 100, 100),
    "khaki": (58, 91, 43),
    "green": (120, 100, 50),
    "mint": (180, 100, 100),
    "skyblue": (203, 46, 98),
    "purple": (300, 100, 50),
    "pink": (330, 59, 100),
    "beige": (36, 79, 77),
}

# 데이터 전처리
color_names = [data["name"] for data in color_data]
hsv_values = torch.tensor([data["hsv"] for data in color_data], dtype=torch.float32)

# Word Embedding 모델
class ColorEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(ColorEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        embedded = self.embeddings(x)
        output = self.linear(embedded)
        return output

# 모델 및 손실 함수, 옵티마이저 정의
vocab_size = len(color_names)
embedding_dim = 10
output_dim = 3  # HSV 값의 차원
model = ColorEmbeddingModel(vocab_size, embedding_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    input_indices = torch.tensor([color_names.index(name) for name in color_names], dtype=torch.long)
    outputs = model(input_indices)
    loss = criterion(outputs, hsv_values)
    loss.backward()
    optimizer.step()




user_sentence = "red top and black bottom"
result = find_color_in_sentence(color_data, user_sentence)

if result:
    print(result)
else:
    print("No matching color found in the sentence.")