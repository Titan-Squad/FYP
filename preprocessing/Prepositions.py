# -*- coding: utf-8 -*-
import csv
if __name__ == '__main__':
    f = open('sinhala_preposition.csv','rt',encoding="utf-8-sig")
    with f:
        data = csv.DictReader(f)
        for row in data:
            print(row['singlish'])