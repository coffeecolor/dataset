import cv2
import numpy as np
import pandas as pd
import csv
import datetime
import time
from sklearn.cluster import MiniBatchKMeans

header = []
fill_header = True
divider = 4


class FColors:
    WARNING = '\033[93m'
    OKGREEN = '\033[92m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def crop_image(img, x, y, h):
    return img[y:y+h, x:x+h]


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


def equalize_hist(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Dividindo a imagem HSV em diferentes canais
    h, s, v = cv2.split(hsv_img)

    # Aplicando equalização de histograma no canal V
    v_equalized = cv2.equalizeHist(v)

    # Unificando os canais H, S e V com equalização aplicada
    hsv_img_equalized = cv2.merge((h, s, v_equalized))

    # Convertendo imagem HSV equalizada em RGB
    img = cv2.cvtColor(hsv_img_equalized, cv2.COLOR_HSV2BGR)
    return img


def color_quantization(img, n_clusters=6):
    # Pega tamanho da imagem
    (h, w) = img.shape[:2]

    # converte a imagem do espaço de cores RGB para o espaço de cores L*a*b*
    # -- já que estaremos agrupando usando k-means # que é baseado na distância euclidiana, usaremos o
    # L*a* b* espaço de cor onde a distância euclidiana implica
    # significado perceptivo
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # remodela a imagem em um vetor para que o k-means possa ser aplicado
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # Aplica o KMeans
    clt = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = clt.fit_predict(img)

    # A imagem reduzida a cores
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # remodela o vetor para imagem novamente
    quant = quant.reshape((h, w, 3))

    # converte de L*a*b* para RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

    # Retorna imagem com cores reduzidas
    return quant


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

        # Suavização pela mediana
        img_coffee = cv2.medianBlur(src=img_coffee, ksize=5)
        img_paper = cv2.medianBlur(src=img_paper, ksize=5)

        # Quantização de cores
        # img_coffee = color_quantization(img_coffee, n_clusters=3)
        # img_paper = color_quantization(img_paper, n_clusters=3)

        # Equalização de histograma
        # img_coffee = equalize_hist(img_coffee)
        # img_paper = equalize_hist(img_paper)

        print('device: {}, flash: {}, agtron: {}'.format(
            row['device_id'], flash_value, agtron_value))

        # Cria um dicionário com os dados da imagem de café (Componentes referentes a cor)
        row_data.extend(mean_std(extract_hsv(img_coffee),
                                 ['H', 'S', 'V'], 'coffee'))
        row_data.extend(mean_std(extract_lab(img_coffee),
                                 ['L', 'A', 'Bl'], 'coffee'))
        row_data.extend([np.mean(extract_grey(img_coffee)),
                         np.std(extract_grey(img_coffee))])
        header.extend(['coffee_grey_mean', 'coffee_grey_std']
                      ) if fill_header else None

        # Cria um dicionário com os dados da imagem de papel (Componentes referentes a iluminação)
        row_data.extend(
            mean_std([extract_hsv(img_paper)[2]], ['V'], 'paper'))
        row_data.extend(
            mean_std([extract_lab(img_paper)[0]], ['L'], 'paper'))

        row_data = [round(num, 3) for num in row_data]
        row_data.extend([flash_value, 'Agtron {}'.format(agtron_value)])
        header.extend(['flash', 'agtron']) if fill_header else None

        data.append(row_data)
        fill_header = False
    return data


def export_csv(data, name):
    with open('./DATA/{}.csv'.format(name), 'w', encoding='UTF8', newline='') as f:
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

    FILE = 'features_' + str(round(time.mktime(ms.timetuple()) * 1000))
    export_csv(get_image_paths(), FILE)

    end_time = time.time()
    print(f"{FColors.BOLD}{FColors.WARNING}{'time = '}{end_time - start_time}{' s'}{FColors.ENDC}\n")
    print(f"{FColors.BOLD}{FColors.OKGREEN}{'--------------------'}{FColors.ENDC}\n")
