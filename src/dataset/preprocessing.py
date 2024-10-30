import numpy as np

def preprocessing(data):
    # TODO : Preprocess the data

    ## =============== BASELINE CODE ================= ##
    # 숫자로 이루어진 행만 추출
    data = data.select_dtypes(include=[np.number])

    # 결측치 0으로 대체
    data = data.fillna(0)

    return data