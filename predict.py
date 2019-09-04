import os
import glob
import jaconv
import pickle
import pandas as pd
import numpy as np
import xlwings as xw
from sklearn.ensemble import RandomForestClassifier
import tkinter
from tkinter import messagebox


#使用するファイル名
home_path = os.getcwd()
predict_folder = os.path.join(home_path, '予測データ')
excel_files = glob.glob(os.path.join(predict_folder, '*.xlsx'))
model = 'model.pickle'
noun_list_n = 'noun_n.pickle'
noun_list_h = 'noun_h.pickle'
noun_list_i = 'noun_i.pickle'


##関数
#全角半角表記揺れの整理の関数
def clean_zh(str):
    str = jaconv.z2h(str, kana=False, digit=True, ascii=False) #数値を半角へ
    str = jaconv.h2z(str, kana=True, digit=False, ascii=True)  #カタカナ、アルファベット、記号を全角へ
    return str

#指定の列に対して、リストの文字列が含まれているかどうか、one-hot化する関数
def add_noun_cols(df, col, list):
    for i in range(len(list)):
        df.at[df[col].str.contains(list[i]), col[0] + str(i)] = 1
        df.at[df[col[0] + str(i)].isnull(), col[0] + str(i)] = 0
    return df


##項目リスト
col_list = ['名称','見出し','明細']
db_cols = ['code','name','header','item']
noun_cols = ['name','header','item']



#Excelを起動しておく
app = xw.App()

#対象のExcelファイルを1つずつ取得する
for sheet_file in excel_files:
    sheet = pd.read_excel(sheet_file, encoding='Shift_JISx0213')

    #名称～明細までで、空の項目があれば''で埋める
    sheet[col_list] = sheet[col_list].fillna('')

    #カタカナと記号は全角、数値は半角に統一
    for i in range(len(sheet)):
        for j in range(len(col_list)):
            sheet.at[i, col_list[j]] = clean_zh(sheet.at[i, col_list[j]])
    
    #学習モデルと単語リストを取得
    with open(os.path.join(home_path, model), mode='rb') as f:
        clf = pickle.load(f)
    with open(os.path.join(home_path, noun_list_n), mode='rb') as f:
        picked_f = pickle.load(f)
    with open(os.path.join(home_path, noun_list_h), mode='rb') as f:
        picked_h = pickle.load(f)
    with open(os.path.join(home_path, noun_list_i), mode='rb') as f:
        picked_i = pickle.load(f)

    #codeを除いたDataFrameを作成
    df_xl = sheet.copy()
    df_xl = df_xl.rename(columns={'コード番号':'code', '名称':'name', '見出し':'header', '明細':'item'})
    df_xl = df_xl.drop('code', axis = 1)

    #各項目をone-hot化
    df_xl = add_noun_cols(df_xl, 'name', picked_n)
    df_xl = add_noun_cols(df_xl, 'header', picked_h)
    df_xl = add_noun_cols(df_xl, 'item', picked_i)
    df_xl = df_xl.drop({'name', 'header', 'item'}, axis = 1)

    pred = clf.predict(df_xl)

    #Excelを開いて、判定したcodeを書き込む
    wb = app.books.open(sheet_file)
    for i in range(len(sheet)):
        xw.Range('A' + str(i + 2)).value = "'" + pred[i]

    #保存して閉じる
    wb.save()
    wb.close()

app.quit()
app.kill()