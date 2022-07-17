import cv2
import numpy as np
import pandas as pd
import csv
import datetime
import time

header = []
fill_header = True


class FColors:
    WARNING = '\033[93m'
    OKGREEN = '\033[92m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def crop_image(img, x, y, h):
    # resized = cv2.resize(img[y:y+h, x:x+h], (512, 512), interpolation = cv2.INTER_AREA)
    return img[y:y+h, x:x+h]


def extract_rgb(img):
    R, G, B = cv2.split(img)
    return (R, G, B)


def extract_hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    return (H, S, V)


def extract_lab(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab_img)
    return (L, A, B)


def extract_grey(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey_img


def mean_std(data, color_space, type):
    list_data = []
    for index, component in enumerate(data):
        list_data.append(np.mean(component))
        header.append('{}_{}_mean'.format(
            type, color_space[index])) if fill_header else None
        list_data.append(np.std(component))
        header.append('{}_{}_std'.format(
            type, color_space[index])) if fill_header else None

    return list_data


def get_image_paths():
    global fill_header
    data = []
    dataframe = pd.read_csv('./photos.csv', delimiter=';')

    for index, row in dataframe.iterrows():
        row_data = []
        img_coffee = crop_image(cv2.imread(
            './RAW/{}'.format(row['name_coffee'])), row['X1'], row['Y1'], row['H'])
        img_paper = crop_image(cv2.imread(
            './RAW/{}'.format(row['name_paper'])), row['X1'], row['Y1'], row['H'])
        agtron_value = row['agtron']
        flash_value = row['flash']

        print('device: {}, flash: {}, agtron: {}'.format(
            row['device_id'], flash_value, agtron_value))

        # Cria um dicionário com os dados da imagem de café (Componentes referentes a cor)
        # row_data.extend(mean_std(extract_rgb(img_coffee),
        #                          ['R', 'G', 'B'], 'coffee'))
        row_data.extend(mean_std(extract_hsv(img_coffee),
                                 ['H', 'S', 'V'], 'coffee'))
        # row_data.extend(mean_std(extract_lab(img_coffee),
        #                          ['L', 'A', 'Bl'], 'coffee'))
        row_data.extend([np.mean(extract_grey(img_coffee)),
                         np.std(extract_grey(img_coffee))])
        header.extend(['coffee_grey_mean', 'coffee_grey_std']
                      ) if fill_header else None

        # Cria um dicionário com os dados da imagem de papel (Componentes referentes a iluminação)
        row_data.extend(
            mean_std([extract_hsv(img_paper)[2]], ['V'], 'paper'))
        row_data.extend(
            mean_std([extract_lab(img_paper)[0]], ['L'], 'paper'))
        row_data.extend([np.mean(extract_grey(img_paper)),
                         np.std(extract_grey(img_paper))])
        header.extend(['paper_grey_mean', 'paper_grey_std']
                      ) if fill_header else None

        row_data = [round(num, 3) for num in row_data]
        row_data.extend([flash_value, 'Agtron {}'.format(agtron_value)])
        header.extend(['flash', 'agtron']) if fill_header else None

        data.append(row_data)
        fill_header = False
    return data


def export_csv(data, name):
    with open('./DATA/DEV/{}.csv'.format(name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # Escreve o cabeçalho (header)
        writer.writerow(header)
        # Escreve todas as linhas (data)
        writer.writerows(data)

        f.close()


if __name__ == '__main__':
    start_time = time.time()
    ms = datetime.datetime.now()

    # warnings.filterwarnings(action='ignore')
    print(f"{FColors.BOLD}{FColors.OKGREEN}{'--------------------'}{FColors.ENDC}\n")

    FILE = 'HSV_features_' + str(round(time.mktime(ms.timetuple()) * 1000))
    export_csv(get_image_paths(), FILE)

    end_time = time.time()
    print(f"{FColors.BOLD}{FColors.WARNING}{'time = '}{end_time - start_time}{' s'}{FColors.ENDC}\n")
    print(f"{FColors.BOLD}{FColors.OKGREEN}{'--------------------'}{FColors.ENDC}\n")
