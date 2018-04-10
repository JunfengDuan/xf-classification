import os
import xlrd


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


def read_xls_file(file_name):
    """
    读取文件并转换为一行（去空格）
    :param file_name:
    :return:
    """
    workbook = xlrd.open_workbook(file_name)
    sheets = workbook.sheet_names()
    worksheet = workbook.sheet_by_name(sheets[0])
    print(file_name, worksheet.name, worksheet.nrows, worksheet.ncols)

    purpose_column_index = get_column_index(worksheet, '信访目的')
    contents_column_index = get_column_index(worksheet, '投诉内容')

    purposes = worksheet.col_values(purpose_column_index)
    contents = worksheet.col_values(contents_column_index)

    purposes_contents = list(zip(purposes, contents))
    items = []
    for item in purposes_contents[1:len(purposes_contents)]:
        clean_content = str(item[1]).replace('\n', '').replace('\t', '').replace('\r', '').replace('  ', '')
        if len(clean_content) < 10:
            continue
        items.append(item[0] + '\t' + clean_content)

    return items


def save_file(dir_name):
    """
    构造训练需要的3个文件
    文件内容格式:  类别\t内容
    :param dir_name: 原数据目录
    :return:
    """
    f_train = open('../datafile/xf_purpose_data/xf_train.txt', 'w', encoding='utf-8')
    f_val = open('../datafile/xf_purpose_data/xf_val.txt', 'w', encoding='utf-8')
    f_test = open('../datafile/xf_purpose_data/xf_test.txt', 'w', encoding='utf-8')

    files = os.listdir(dir_name)
    for file in files:
        filename = os.path.join(dir_name, file)
        purpose_content = read_xls_file(filename)

        count = 0
        for text in purpose_content:
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
    create_file('../datafile/xf_purpose_data')
    save_file('../datafile/xf_data')
    print(len(open('../datafile/xf_purpose_data/xf_train.txt', 'r', encoding='utf-8').readlines()))  # 60000
    print(len(open('../datafile/xf_purpose_data/xf_val.txt', 'r', encoding='utf-8').readlines()))  # 6000
    print(len(open('../datafile/xf_purpose_data/xf_test.txt', 'r', encoding='utf-8').readlines()))  # 10216

