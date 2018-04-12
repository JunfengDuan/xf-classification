import os
import xlrd
import numpy as np


def read_yy_file(file_path):
    yy_words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            yy_words.append(line)
    return yy_words


def create_file(dir_name):
    """
    创建语料空文件：训练集、验证集、测试集
    :param dir_name:
    :return:
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    xf_train_file = os.path.join(dir_name, 'xf_train.txt')
    xf_val_file = os.path.join(dir_name, 'xf_val.txt')
    xf_test_file = os.path.join(dir_name, 'xf_test.txt')

    if os.path.exists(xf_train_file):
        os.remove(xf_train_file)
    os.mknod(xf_train_file)
    if os.path.exists(xf_val_file):
        os.remove(xf_val_file)
    os.mknod(xf_val_file)
    if os.path.exists(xf_test_file):
        os.remove(xf_test_file)
    os.mknod(xf_test_file)


def get_column_index(table, column_name):
    column_index = None
    for i in range(table.ncols):
        if table.cell_value(0, i) == column_name:
            column_index = i
            break
    return column_index


def read_xls_file(file_name, yy_words):
    """
    读取文件并转换为一行（去空格）
    :param file_name:
    :return:
    """
    workbook = xlrd.open_workbook(file_name)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[0])
    print(file_name, worksheet.name, worksheet.nrows, worksheet.ncols)

    yy_column_index = get_column_index(worksheet, '是否扬言')
    contents_column_index = get_column_index(worksheet, '投诉内容')

    yy = worksheet.col_values(yy_column_index)
    contents = worksheet.col_values(contents_column_index)

    yy_contents = list(zip(yy, contents))

    items = yy_content_process(yy_contents, yy_words)

    return items


def yy_content_process(yy_contents, yy_words):
    """
    扬言插入正例样本
    :param yy_words:
    :param yy_contents:
    :return:
    """

    items = []
    for item in yy_contents[1:len(yy_contents)]:
        clean_content = str(item[1]).replace('\n', '').replace('\t', '').replace('\r', '').replace('  ', '')
        tag = item[0]
        if len(clean_content) < 10:
            continue
        rand = np.random.randint(1, 10)  # 将语料库中三分之一的样本改造成正例样本，剩余为负样本
        if rand % 3 == 0:
            yy_1 = yy_words[np.random.randint(0, 12)]
            yy_2 = yy_words[np.random.randint(0, 12)]

            tag = '是'
            clean_content = clean_content.replace('，', '，;,$,').replace(';', yy_1).replace('$', yy_2)
        items.append(tag + '\t' + clean_content)

    return items


def save_file(dir_name):
    """
    构造训练需要的3个文件
    文件内容格式:  类别\t内容
    :param dir_name: 原数据目录
    :return:
    """
    f_train = open('../datafile/xf_yy_data/xf_train.txt', 'w', encoding='utf-8')
    f_val = open('../datafile/xf_yy_data/xf_val.txt', 'w', encoding='utf-8')
    f_test = open('../datafile/xf_yy_data/xf_test.txt', 'w', encoding='utf-8')

    yy_words = read_yy_file('../datafile/yy_word.txt')
    print('扬言敏感词：', yy_words)

    files = os.listdir(dir_name)
    for file in files:
        filename = os.path.join(dir_name, file)
        yy_content = read_xls_file(filename, yy_words)

        count = 0
        for text in yy_content:
            if count < 20000:
                f_train.write(text + '\n')
            elif count < 22000:
                f_val.write(text + '\n')
            elif count < 27000:
                f_test.write(text + '\n')
            else:
                break
            count += 1

    f_train.close()
    f_val.close()
    f_test.close()
    print('Write finished')

if __name__ == '__main__':
    create_file('../datafile/xf_yy_data')
    save_file('../datafile/xf_data')
    print(len(open('../datafile/xf_yy_data/xf_train.txt', 'r', encoding='utf-8').readlines()))  # 60000
    print(len(open('../datafile/xf_yy_data/xf_val.txt', 'r', encoding='utf-8').readlines()))  # 6000
    print(len(open('../datafile/xf_yy_data/xf_test.txt', 'r', encoding='utf-8').readlines()))  # 10216/10219

