import json

if __name__ == '__main__':
    confusion_matrix = [[0 for _ in range(5)] for _ in range(5)]
    with open("results\\test.all.txt", encoding="UTF-8") as f:
        for line in f:
            case = json.loads(line)
            prediction = int(float(case["PD"]))
            groud_truth = int(float(case["GT"]))
            confusion_matrix[prediction][groud_truth] += 1

    for i in range(5):
        print('\n')
        print('\t'.join(str(i) for i in confusion_matrix[i]))