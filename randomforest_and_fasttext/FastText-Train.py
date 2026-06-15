import fasttext

train_data_path='./data/train_fast.txt'
test_data_path='./data/test_fast.txt'
dev_data_path='./data/dev_fast.txt'
model = fasttext.train_supervised(train_data_path,
                                  wordNgrams=2,
                                  autotuneValidationFile=dev_data_path,
                                  verbose=3,
                                  thread=10)
result=model.test(test_data_path)
print(result)
model.save_model('./model/fasttext_model.bin')