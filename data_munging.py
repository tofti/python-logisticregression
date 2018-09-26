import ast
import csv
import os
import copy


def get_header_name_to_idx_maps(headers):
    name_to_idx = {}
    idx_to_name = {}
    for i in range(0, len(headers)):
        name_to_idx[headers[i]] = i
        idx_to_name[i] = headers[i]
    return idx_to_name, name_to_idx


def replace_str_with_float(list_of_list):
    for l in list_of_list:
        for idx in range(0, len(l)):
            try:
                f = float(l[idx])
                l[idx] = f
            except ValueError:
                pass


def load_csv_to_header_data(filename):
    fpath = os.path.join(os.getcwd(), filename)
    fs = csv.reader(open(fpath, newline='\n'))
    all_row = []
    for r in fs:
        all_row.append(r)

    headers = all_row[0]
    idx_to_name, name_to_idx = get_header_name_to_idx_maps(headers)

    data = {
        'header': headers,
        'rows': all_row[1:],
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name
    }
    replace_str_with_float(data['rows'])
    return data


def project_columns(data, columns_to_project):
    data_h = list(copy.copy(data['header']))
    data_r = list(copy.deepcopy(data['rows']))

    all_cols = list(range(0, len(data_h)))

    columns_to_project_ix = [data['name_to_idx'][name] for name in columns_to_project]
    columns_to_remove = [cidx for cidx in all_cols if cidx not in columns_to_project_ix]

    for delc in sorted(columns_to_remove, reverse=True):
        del data_h[delc]
        for r in data_r:
            del r[delc]

    idx_to_name, name_to_idx = get_header_name_to_idx_maps(data_h)

    return {'header': data_h,
            'name_to_idx': name_to_idx,
            'idx_to_name': idx_to_name,
            'rows': data_r}


def load_config(config_file):
    with open(config_file, 'r') as myfile:
        data = myfile.read().replace('\n', '')
    return ast.literal_eval(data)


def fill(x, n):
    l = []
    for i in range(n):
        l.append(x)
    return l