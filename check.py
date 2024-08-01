import os

model_path_1 = '../widar3/val_location/v2_2_T2_S_r3/model_best.pth.tar'
model_path_2 = '../widar3/val_location//v2_2_T3_S_r1/checkpoint_current.pth.tar'

size_1 = os.path.getsize(model_path_1)
size_2 = os.path.getsize(model_path_2)

print(f"Model 1 size: {size_1} bytes")
print(f"Model 2 size: {size_2} bytes")

if size_1 == size_2:
    print("The model sizes are identical.")
else:
    print("The model sizes are different.")
