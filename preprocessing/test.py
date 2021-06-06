import json
if __name__ == '__main__':
    with open("UniqueWords.txt", 'r', encoding='utf-8', errors='ignore') as outfile:
        newslist = json.load(outfile)
        print(newslist)
