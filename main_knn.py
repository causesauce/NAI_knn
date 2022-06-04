import matplotlib.pyplot as plt


def key_value(val, dictionary):
    for key, value in dictionary.items():
        if value == val:
            return key
    return -1


def translate_and_categorize(file_path, labels_dict=None):
    if labels_dict is None:
        f = open(file_path)
        labels_dict = dict()
        dataset = list()
        number = 1
        for i in f:
            tmp_data_row, tmp_label = i.split(',')[:-1], i.split(',')[-1].strip()
            if tmp_label not in labels_dict.keys():
                labels_dict[tmp_label] = float(number)
                number += 1
            tmp_data_row = [float(j) for j in tmp_data_row]
            tmp_data_row.append(float(labels_dict[tmp_label]))
            dataset.append(tmp_data_row)

        f.close()
        return labels_dict, dataset
    else:
        f = open(file_path)
        dataset = list()
        for i in f:
            tmp_data_row, tmp_label = i.split(',')[:-1], i.split(',')[-1].strip()
            tmp_data_row = [float(j) for j in tmp_data_row]
            tmp_data_row.append(float(labels_dict[tmp_label]))
            dataset.append(tmp_data_row)

        f.close()
        return dataset


def process_knn(training_path, test_path, k):
    if k < 1:
        return -1
    else:
        training_labels_dict, training_dataset = translate_and_categorize(training_path)
        test_dataset = translate_and_categorize(test_path, labels_dict=training_labels_dict)
        result = list()
        counter = 0

        for i in test_dataset:
            trace = list()
            for j in training_dataset:
                distance = sum((d - l) ** 2 for d, l in zip(i[0:-1], j[0:-1])) ** 0.5
                trace.append([distance, j[-1]])
            trace.sort(key=lambda x: x[0])
            trace = trace[:k]
            counter += 1
            result.append([counter, i[-1], [j[-1] for j in trace]])

        count_dict = dict.fromkeys(training_labels_dict.values(), 0)
        prediction = list()

        for i in result:
            for j in i[-1]:
                count_dict[j] += 1
            prediction.append([i[0], i[1], key_value(max(count_dict.values()), count_dict)])
            count_dict = dict.fromkeys(training_labels_dict.values(), 0)

        counter = 0

        for i in prediction:
            if i[1] == i[2]:
                counter += 1

        result = round(counter / len(test_dataset), 3) * 100
        print(result, '% accuracy')
        return result


def process_one_row(training_path, data_row, k):
    if k < 0:
        return -1
    else:
        training_labels_dict, training_dataset = translate_and_categorize(training_path)
        data_row = data_row.split(',')
        data_row = [float(i) for i in data_row]
        trace = list()
        counter = 1

        for j in training_dataset:
            distance = sum((d - l) ** 2 for d, l in zip(data_row, j[0:-1])) ** 0.5
            trace.append([distance, j[-1]])
            counter += 1
        trace.sort(key=lambda x: x[0])

        trace = trace[:k]
        count_dict = dict.fromkeys(training_labels_dict.values(), 0)

        for i in trace:
            count_dict[i[-1]] += 1

        print('your record is predicted to be'
              , key_value(key_value(max(count_dict.values())
                                    , count_dict)
                          , training_labels_dict)
              )

# plot accuracy vs K



# %%
# main method for starting the program
if __name__ == '__main__':
    training_file_path = input('provide training file path: ')
    k = input('provide number k: ')
    input_parameter = input('provide 0 if you want to estimate your own row: ')
    if int(input_parameter) == 0:
        data_row = input('provide data row: ')
        process_one_row(training_file_path, data_row, int(k))
    else:
        test_file_path = input('provide test file path: ')
        process_knn(training_file_path, test_file_path, int(k))


#%%
training_file_path = 'train.txt'
test_file_path = 'test.txt'
result_x = list()
result_y = list()
for i in range(15):
    result_x.append(i)
    result_y.append(process_knn(training_file_path, test_file_path, i))
plt.xlabel('k-number')
plt.ylabel('accuracy in %')
plt.plot(result_x, result_y, 'ro')
plt.axis([-2, 15, -2, 110])
plt.show()


