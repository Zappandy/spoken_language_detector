# example, tst data has 540 files

test_data = SpeechDataset(test_dir, "librosa")
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

train_data = SpeechDataset(train_dir, "librosa")
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)


#TODO  finish up the splits
train_data = Subset(train_data, torch.arange(240)) # 80% 
test_data = Subset(test_data, torch.arange(60))  # 20%

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

# test visualizations

melspec_test = next(iter(test_dataloader))
print(melspec_test)
