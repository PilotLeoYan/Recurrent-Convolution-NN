import data


filepaths = data.download_moving_mnist()
print(filepaths)

aug = data.transform()

train_ds = data.MovingMNISTDataset(filepaths[0], transform=aug)
valid_ds = data.MovingMNISTDataset(filepaths[1])
test_ds = data.MovingMNISTDataset(filepaths[2])
print(f'len(train_ds): {len(train_ds)}')
print(f'len(valid_ds): {len(valid_ds)}')
print(f'len(test_ds): {len(test_ds)}')

print(f'train_ds[1].shape: {train_ds[1].shape}')
print(f'valid_ds[2].shape: {valid_ds[2].shape}')
print(f'test_ds[3].shape: {test_ds[3].shape}')

train_loader = data.make_dataloader(train_ds, 32, train=True)
valid_loader = data.make_dataloader(valid_ds, 64)
test_loader = data.make_dataloader(test_ds, 64)

# del train_ds, valid_ds, test_ds

for loader in (train_loader, valid_loader, test_loader):
    batch = next(iter(loader))
    print(batch.shape)
