import json

if __name__ == '__main__':
    confusion_matrix = [[0 for _ in range(5)] for _ in range(5)]
    mse = 0
    cnt = 0
    with open("results\\test.all.txt", encoding="UTF-8") as f:
        for line in f:
            case = json.loads(line)
            prediction = int(float(case["PD"]))
            groud_truth = int(float(case["GT"]))
            confusion_matrix[prediction][groud_truth] += 1
            cnt += 1
            mse += (prediction - groud_truth) * (prediction - groud_truth)

    for i in range(5):
        print('\n')
        print('\t'.join(str(i) for i in confusion_matrix[i]))
    mse /= cnt
    print(mse)