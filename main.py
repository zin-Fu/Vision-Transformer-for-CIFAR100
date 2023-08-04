from train import *
from val import *

print("😁 Vision Transformer for CIFAR-100")
get_and_show_pictures(train_dataset, std, mean)

print("🚀 Building model...\n")
model = get_model()
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
print("Successfully build!\n")

print("🚀 Training model on", DEVICE)
train(model, optimizer=optimizer)


print("\nEvaluating...")
evaluation_and_show(model)
