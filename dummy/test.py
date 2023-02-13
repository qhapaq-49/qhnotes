from dlshogi.common import *
import torch
import torch.nn as nn
from policy_value_network_qhapaq import PolicyValueNetwork
from dlshogi.data_loader import DataLoader

# gpuじゃなくても数分で計算は終わる
device = torch.device("cpu")

# ネットワークに設定(senet15layer+224channel。評価関数によってここは変わる)
model = PolicyValueNetwork(blocks=15, channels=224, activation=nn.SiLU())
model.to(device)

# pthファイルを読み込む
checkpoint = torch.load("dummy.pth", map_location=device)
model.load_state_dict(checkpoint)
print("load model OK")


# dlshogiのtrain.pyから持ってきたベンチマーク用の各種関数
def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)


test_data = np.fromfile("bench.hcpe", dtype=HuffmanCodedPosAndEval)
test_dataloader = DataLoader(test_data, 128, device)


# benchmark
steps = 0
sum_test_loss1 = 0
sum_test_loss2 = 0
sum_test_loss3 = 0
sum_test_loss = 0
sum_test_accuracy1 = 0
sum_test_accuracy2 = 0
sum_test_entropy1 = 0
sum_test_entropy2 = 0

cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

model.eval()

for x1, x2, t1, t2, value in test_dataloader:
    y1, y2 = model(x1, x2)
    steps += 1
    loss1 = cross_entropy_loss(y1, t1).mean()
    loss2 = bce_with_logits_loss(y2, t2)
    loss3 = bce_with_logits_loss(y2, value)
    loss = loss1 + (1 - 0.33) * loss2 + 0.33 * loss3
    sum_test_loss1 += loss1.item()
    sum_test_loss2 += loss2.item()
    sum_test_loss3 += loss3.item()
    sum_test_loss += loss.item()
    sum_test_accuracy1 += accuracy(y1, t1)
    sum_test_accuracy2 += binary_accuracy(y2, t2)

# dummy.pthのベンチマーク
# test loss = 2.0078341, 0.5362287, 0.6087358, 2.5679901, test accuracy = 0.3839286, 0.7176339
print(f'test loss = {sum_test_loss1/steps:.07f}, {sum_test_loss2/steps:.07f}, {sum_test_loss3/steps:.07f}, {sum_test_loss/steps:.07f}, test accuracy = {sum_test_accuracy1/steps:.07f}, {sum_test_accuracy2/steps:.07f}')
