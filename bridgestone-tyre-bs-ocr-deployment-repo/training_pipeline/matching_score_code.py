import argparse
import re



# def regexMatcher(actural_string,predicted_string, N=4):
#     # if len(predicted_string) > len(actural_string):
#     #     return 'No'
#     no_of_detected_chars = len(predicted_string)-predicted_string.count('_')
#     if no_of_detected_chars < N:
#         return 'No'
#     p1 = predicted_string.replace('_','[A-Z,0-9]') #+ one or more #add + or not
#     p2 = '[A-Z,0-9]*'+p1
#     re.search(p1,actural_string)
#     if re.search(p1,actural_string):
#         print(re.search(p1,actural_string))
#         return 'Yes'
#     elif re.search(p2,actural_string):
#         return 'Yes'
#     else:
#         return 'No'


def regexMatcher(actural_string,predicted_string, N=5):
    # no_of_detected_chars = len(predicted_string)-predicted_string.count('-')
    # if no_of_detected_chars < N:
        # return False
    p1 = predicted_string.replace('-','[A-Z,0-9]') #+ one or more #add + or not
    p2 = '[A-Z,0-9]*'+p1
    re.search(p1,actural_string)
    if re.search(p1,actural_string):
        print(re.search(p1,actural_string))
        return True
    elif re.search(p2,actural_string):
        return True
    else:
        return False


def str_matcher(tag_gt, tag_pred, N=4):
    ind = -1
    count = 0
    for c in tag_pred:
        if c == "-":
            continue
        ind = tag_gt.find(c, ind+1)
        if ind != -1:
            count += 1
    if count > N:
        return "Yes"
    return "No"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--master_tyre_number", help = "",required=True)
    parser.add_argument("-p", "--predicted_tyre_number", help = "",required=True)
    args = parser.parse_args()
    print(regexMatcher(args.master_tyre_number, args.predicted_tyre_number))