#F0D5F1

def events_from_list(input_list):
    return [num for num in input_list if num % 2 == 0]


def every_element_is_odd(input_list):
    if len([num for num in input_list if num % 2 != 0]) == len(input_list): return True
    else:
        return False


def kth_largest_in_list(input_list, kth_largest):
    list2 = list(set(input_list))
    list2.sort()
    return (list2[-kth_largest])


def cumavg_list(input_list):
    cum_sum = 0
    res = []
    for index, value in enumerate(input_list, start=1):
        cum_sum += value
        cum_avg = cum_sum / index
        res.append(cum_avg)
    return res


def element_wise_multiplication(input_list1, input_list2):
    res = [a * b for a, b in zip(input_list1, input_list2)]
    return res


def merge_lists(*lists):
    merged = []
    for i in lists:
        merged.extend(i)
    return merged


def squared_odds(input_list):
    odds = [num for num in input_list if num % 2 != 0]
    res = [x ** 2 for x in odds]
    return res


def reverse_sort_by_key(input_dict):
    keys = list(input_dict.keys())
    keys.sort(reverse = True)
    sorted_dict = {i:input_dict[i] for i in keys}
    return sorted_dict


def sort_list_by_divisibility(input_list):
    res = {
        "by_two": [i for i in input_list if i % 2 == 0],
        "by_five": [i for i in input_list if i % 5 == 0],
        "by_two_and_five": [i for i in input_list if i % 2 == 0 and i % 5 == 0],
        "by_none": [i for i in input_list if i % 2 == 1 and i % 5 != 0 ]
    }
    return res


