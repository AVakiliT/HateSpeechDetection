import pandas
path = 'Dataset/Sarcasm/'

def clean(file_name):
    df = pandas.read_csv(path + file_name + '.txt', delimiter='\t', names=['x', 'label', 'text'])
    df.drop(columns=['x'], inplace=True)
    df = df[['text', 'label']]

    df.to_csv(path + file_name + '.csv', index=False)

clean('train')
clean('test')