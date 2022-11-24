import nltk
import pandas as pd
import csv


def main():
    # nltk.download('punkt')
    # nltk.download('stopwords')
    df = pd.read_csv("info/reviews.csv", usecols=[4])
    stop_words = set()
    with open(f'info/stop_words.txt', 'r+', encoding='utf-8') as f:
        for line in f:
            s = str(line).strip()
            # print(s)
            stop_words.add(s)
    # print(stop_words)
    content = df.to_string()
    words = nltk.word_tokenize(content)
    word_list = [w.strip().lower() for w in words]
    filter_words = [word for word in word_list if word.isalpha() and word not in stop_words]
    print(filter_words)
    freq_dist = nltk.FreqDist(filter_words)
    common_words = freq_dist.most_common(50)
    print(common_words)
    with open(f'info/freq_words.csv', 'a+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for i in common_words:
            writer.writerow(i)


if __name__ == "__main__":
    main()
