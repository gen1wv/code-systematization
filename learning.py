import os
import glob
import MeCab
import jaconv
import shutil
import pickle
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np


#使用するファイル名
home_path = os.getcwd()
db_file = 'codes.db'
correct_folder = os.path.join(home_path, '正解データ')
old_folder = os.path.join(correct_folder, '学習済み')
excel_files = glob.glob(os.path.join(correct_folder, '*.xlsx'))
model = 'model.pickle'
noun_list_n = 'noun_n.pickle'
noun_list_h = 'noun_h.pickle'
noun_list_i = 'noun_i.pickle'

col_list = ['名称','見出し','明細']
db_cols = ['code','name','header','item']
noun_cols = ['name','header','item']

#全角半角表記揺れの整理の関数
def clean_zh(str):
    str = jaconv.z2h(str, kana=False, digit=True, ascii=False) #数値を半角へ
    str = jaconv.h2z(str, kana=True, digit=False, ascii=True)  #カタカナ、アルファベット、記号を全角へ
    return str


#積み上げ用DataFrame
stack = pd.DataFrame(columns=db_cols)

for sheet_file in excel_files:
    sheet = pd.read_excel(sheet_file, encoding='Shift_JISx0213',
                          dtype={'コード番号':str, '名称':str, '見出し':str, '明細':str})

    #名称～明細までで、空の項目があれば''で埋める
    sheet[col_list] = sheet[col_list].fillna('')

    #カタカナと記号は全角、数値は半角に統一
    for i in range(len(sheet)):
        for j in range(len(col_list)):
            sheet.at[i, col_list[j]] = clean_zh(sheet.at[i, col_list[j]])

    #Excel上の列名を変換
    sheet = sheet.rename(columns={'コード番号':'code', '名称':'name', '見出し':'header', '明細':'item'})

    #積み上げ
    stack = stack.append(sheet)

    #正解データ下のExcelファイルを、学習済みフォルダーへ移動
    shutil.move(sheet_file), old_folder)

stack = stack.reset_index(drop=True)



#DBへ登録し、重複削除したDataFrameを取得
with sqlite3.connect(os.path.join(home_path, db_file)) as conn:
    stack.to_sql('newdata', conn, if_exists='replace', index=None)
    conn.execute('delete from learned ld where exists (select 1 from newdata nd where ld.code   = nd.code   and ' +
                                                                                    ' ld.name   = nd.name   and ' +
                                                                                    ' ld.header = nd.header and ' +
                                                                                    ' ld.item   = nd.item)')
    stack.to_sql('learned', conn, if_exists='append', index=None)
    stack = pd.read_sql_query('select * from learned', conn)



##学習モデルの作成
#相関係数の設定
corr_rate = 0.2

#one-hot用の関数
def one_hot(df, col):
    buf = pd.get_dummies(df[col])
    df = pd.concat([df, buf], axis=1)
    df = df.drop(col, axis=1)
    return df

#形態素解析して、名詞だけリスト化する関数
def make_noun_list(df, col):
    mt = MeCab.Tagger()
    list = []
    for i in range(0, len(df)):
        node = mt.parseToNode(df.at[i, col])
        while node is not None:
            hinshi = node.feature.split(",")
            if hinshi[0] != 'BOS/EOS':
                if hinshi[0] == '名詞' and not hinshi[1] in ['固有名詞', '数'] and hinshi[6] != '*':
                    if not node.surface in list:
                        list.append(node.surface)
            node = node.next
    return list

#相関の強い名詞一覧を抽出してリスト化する関数
def pick_noun_list(df_org, col, list, codes, corr_rate, noun_cols):
    #各項目で単語を抽出してone-hot化
    df = df_org.copy()
    df = df[[noun_cols, col]]
    for i in range(len(list)):
        df.at[df[col].str.contains(list[i]), col[0] + str(i)] = 1
        df.at[df[col[0] + str(i)].isnull(), col[0] + str(i)] = 0

    #one-hot化し、codeに対する相関係数を算出
    df = one_hot(df, noun_cols)
    cr = df.corr()
    cr = cr[codes]

    #codeと相関の強い単語を抽出
    cr = cr.drop(codes, axis = 0)

    cr2 = pd.DataFrame()
    for i in range(0,len(codes)):
        if i==0:
            cr2 = cr[(cr[codes[i]]>corr_rate)|(cr[codes[i]]<-corr_rate)]
        else:
            cr2 = pd.concat([cr2, cr[(cr[codes[i]]>corr_rate)|(cr[codes[i]]<-corr_rate)]], axis=0)
    cr2 = cr2.reset_index(drop=False)
    cr2 = cr2.drop_duplicates()
    cr2 = cr2.set_index('index')

    #相関の強い名詞一覧を作成
    picked = []
    for i in range(len(cr2)):
        picked.append(list[int(cr2.index[i].replace(col[0], ''))])
    return picked

#指定の列に対して、リストの文字列が含まれているかどうか、one-hot化する関数
def add_noun_cols(df, col, list):
    for i in range(len(list)):
        df.at[df[col].str.contains(list[i]), col[0] + str(i)] = 1
        df.at[df[col[0] + str(i)].isnull(), col[0] + str(i)] = 0
    return df

#関数を呼び出して、リストを生成
list_n = make_noun_list(stack, 'name')
list_h = make_noun_list(stack, 'header')
list_i = make_noun_list(stack, 'item')

#codeをリスト化
stack_c = stack.copy()
stack_c = stack_c.drop_duplicates(subset=target).sort_values(target)
codes = stack_c[target].values.tolist()

#相関の強い名詞リストを生成
picked_n = pick_noun_list(stack, 'name', list_n, codes, corr_rate, target)
picked_h = pick_noun_list(stack, 'header', list_h, codes, corr_rate, target)
picked_i = pick_noun_list(stack, 'item', list_i, codes, corr_rate, target)

#相関の強い名詞だけを項目追加して、one-hot化する
stack_oh = stack.copy()
stack_oh = add_noun_cols(stack_oh, 'name', picked_n)
stack_oh = add_noun_cols(stack_oh, 'header', picked_h)
stack_oh = add_noun_cols(stack_oh, 'item', picked_i)
stack_oh = stack_oh.drop(noun_cols, axis = 1)

#学習用データの生成
x = stack_oh.drop(target, axis = 1)
y = stack_oh[target]

#RandomForestによる学習モデルの生成
clf = RandomForestClassifier()
clf.fit(x, y)

#学習モデルと名詞リストを保存
with open(os.path.join(home_path, model), mode='wb') as f:
    pickle.dump(clf, f)
with open(os.path.join(home_path, noun_list_n), mode='wb') as f:
    pickle.dump(picked_n, f)
with open(os.path.join(home_path, noun_list_h), mode='wb') as f:
    pickle.dump(picked_h, f)
with open(os.path.join(home_path, noun_list_i), mode='wb') as f:
    pickle.dump(picked_i, f)