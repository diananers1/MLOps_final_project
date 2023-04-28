import pandas as pd

if __name__ == "__main__":
    data_url = 'https://drive.google.com/file/d/1ZPZUAlXFEA33Mcm_wgOQDmujmG39j7zQ/view?usp=sharing'
    data_url = 'https://drive.google.com/uc?id=' + data_url.split('/')[-2]
    data = pd.read_csv(data_url)

    test_url = "https://drive.google.com/file/d/1zPyFVxax1_TN7gJ7WGeHt4am31skdIGI/view?usp=sharing"
    test_url = "https://drive.google.com/uc?id=" + test_url.split('/')[-2]
    test = pd.read_csv(test_url)


    data.to_csv("gb_classifier/datasets/credit_score.csv")
    test.to_csv("gb_classifier/datasets//test.csv")

    print("Data successfully uploaded!")