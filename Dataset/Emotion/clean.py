import pandas


path = 'Dataset/Emotion/'
file_name = 'train'

def clean(file_name):
    xs = []
    ys = []
    with open(path + file_name + '.txt', encoding='utf8') as f:
        s = f.readlines()
        del s[0]
        for i in s:
            us = i.split('\t')
            x = ' '.join(us[1:-1]).strip()
            y = us[-1].strip()
            xs.append(x)
            k = {
                'happy': 0,
                'sad': 1,
                'angry': 2,
                'others': 3
            }
            ys.append(k[y])

        df = pandas.DataFrame.from_dict({'text': xs, 'label': ys})
        df.to_csv(path + file_name + '.csv', index=False)

clean('train')
clean('test')