import pandas as pd


def main():
    videos_list = ['believe', 'call', 'can',
                   'chat', 'do', 'happen',
                   'happy', 'hat', 'have',
                   'help', 'I', 'idea',
                   'it', 'next', 'no',
                   'see', 'time', 'understand',
                   'what', 'you',
                   ]

    d1 = {}
    d2 = {}

    for i in range(len(videos_list)):
        d1[i] = [videos_list[i]]
        d2[videos_list[i]] = [i]

    df1 = pd.DataFrame.from_dict(d1)
    df2 = pd.DataFrame.from_dict(d2)

    df1.to_csv('itos.csv', index=False)
    df2.to_csv('stoi.csv', index=False)

    print('Done!')


if __name__ == "__main__":
    main()
