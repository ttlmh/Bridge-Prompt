import torch
import clip
import numpy as np


def text_prompt_slide(classes, id_list, dataset, cnt_max=5):
    text_aug_cnts = [f"This clip contains no actions.",
                     f"This clip contains only one action,", f"This clip contains two actions,",
                     f"This clip contains three actions,", f"This clip contains four actions,",
                     f"This clip contains five actions,", f"This clip contains six actions,",
                     f"This clip contains seven actions,", f"This clip contains eight actions,"]
    text_aug_acts = [f"Firstly, ", f"Secondly, ", f"Thirdly, ", f"Fourthly, ",
                     f"Fifthly, ", f"Sixthly, ", f"Seventhly, ", f"Eighthly, "]
    text_aug_temp = [f"the person is {{}}.", f"the person is performing the action of {{}}.",
                     f"the character is {{}}.", f"he or she is {{}}.", f"the action {{}} is being played.",
                     f"it is the action of {{}}.", f"the human is {{}}.",
                     f"the person is working on {{}}.", f"the scene is {{}}.",
                     f"the person is focusing on {{}}.", f"the person is completing the action of {{}}.",
                     f"the step is {{}}", f"the action is {{}}.", f"the action step is {{}}."]
    text_long_temp = [f"the person is {{}}.", f"the character is {{}}.", f"he or she is {{}}.",
                      f"the human is {{}}.", f"the scene is {{}}.", f"{{}} is being done.",
                      f"the step is {{}}", f"the action is {{}}.", f"the action step is {{}}."]
    text_no_acts = [f"The first action does not exist.",
                    f"The second action does not exist.", f"The third action does not exist.",
                    f"The fourth action does not exist.", f"The fifth action does not exist.",
                    f"The sixth action does not exist.", f"The seventh action does not exist.",
                    f"The eighth action does not exist."]
    text_aug_cnts = text_aug_cnts[:cnt_max+1]
    text_aug_acts = text_aug_acts[:cnt_max]
    text_no_acts = text_no_acts[:cnt_max]

    b, _ = id_list.shape
    num_temp = len(text_aug_temp)
    num_long = len(text_long_temp)
    text_id = np.random.randint(num_temp, size=len(id_list) * cnt_max).reshape(-1, cnt_max)
    text_id_long = np.random.randint(num_long, size=len(id_list) * cnt_max).reshape(-1, cnt_max)
    id_list_cnt = id_list >= 0
    id_list_cnt = torch.sum(id_list_cnt, dim=1)
    res_token_cnt = []

    for id in id_list_cnt:
        res_token_cnt.append(clip.tokenize(text_aug_cnts[id.item()]))
    res_token_cnt = torch.cat(res_token_cnt)

    res_token_acts = []
    res_token_all = []

    for ii, txt in enumerate(id_list):
        num_acts = id_list_cnt[ii].item()
        action_list = []
        for i in range(num_acts):
            action_list.append(classes[txt[i].item()])
        if dataset == 'breakfast':
            if action_list[0] == 'SIL': action_list[0] = 'waiting and preparing'
            if action_list[-1] == 'SIL': action_list[-1] = 'finishing and waiting'
        sentences = []
        sentences_all = ''
        for i in range(num_acts):
            sent = text_aug_acts[i] + text_aug_temp[text_id[ii][i]].format(action_list[i])
            sentences.append(clip.tokenize(sent))
            sentences_all += ' ' + text_aug_acts[i] + text_long_temp[text_id_long[ii][i]].format(action_list[i])
        for i in range(num_acts, len(text_no_acts)):
            sentences.append(clip.tokenize(text_no_acts[i]))
        res_token_acts.append(torch.cat(sentences))
        sentences_all = sentences_all[1:]
        res_token_all.append(clip.tokenize(sentences_all))
        # token = clip.tokenize(sentence)
        # res_token.append(token)
        # text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])
    res_token_acts = torch.cat(res_token_acts).view(b, -1, res_token_cnt.shape[1])
    res_token_all = torch.cat(res_token_all)
    # classes = torch.cat([v for k, v in text_dict.items()])

    return res_token_cnt, res_token_acts, res_token_all, id_list_cnt


def text_prompt_pos_emb():
    num_max = 8
    text_aug_cnts = [f"This clip contains no actions.",
                     f"This clip contains only one action,", f"This clip contains two actions,",
                     f"This clip contains three actions,", f"This clip contains four actions,",
                     f"This clip contains five actions,", f"This clip contains six actions,",
                     f"This clip contains seven actions,", f"This clip contains eight actions,"]
    text_aug_acts = [f"this is the first action.", f"this is the second action.",
                     f"this is the third action.", f"this is the fourth action.",
                     f"this is the fifth action.", f"this is the sixth action.",
                     f"this is the seventh action.", f"this is the eighth action."]
    text_aug_no = "This action does not exist."

    text_dict_acts = {}

    for ii, txt in enumerate(text_aug_cnts):
        lst = []
        for i in range(num_max):
            if i >= ii:
                lst.append(clip.tokenize(text_aug_no))
            else:
                lst.append(clip.tokenize(text_aug_cnts[ii] + ' ' + text_aug_acts[i]))
        text_dict_acts[ii] = lst
        text_dict_acts[ii] = torch.cat(text_dict_acts[ii])
    text_dict_acts = torch.cat([v for k, v in text_dict_acts.items()])

    return text_dict_acts


def text_prompt_ord_emb(cnt_max=5):
    text_aug_acts = [f"this is the first action.", f"this is the second action.",
                     f"this is the third action.", f"this is the fourth action.",
                     f"this is the fifth action.", f"this is the sixth action.",
                     f"this is the seventh action.", f"this is the eighth action."]
    text_aug_acts = text_aug_acts[:cnt_max]

    lst = [clip.tokenize(txt) for txt in text_aug_acts]
    lst = torch.cat(lst)

    return lst


def text_prompt_single(data):
    text_aug = [f"the person is {{}}", f"the person is performing the activity of {{}}",
                f"the character is {{}}", f"he or she is {{}}", f"the human activity of {{}} is being performed",
                f"this video is the activity of {{}}", f"the human is {{}}",
                f"Can you recognize the activity of {{}}?"]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.items()])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict


# def text_prompt_slide_val(classes):
#     text_aug_cnts = [f"This clip contains no actions.",
#                      f"This clip contains only one step.", f"This clip contains two steps.",
#                      f"This clip contains three steps.", f"This clip contains four steps.",
#                      f"This clip contains five steps."]
#     text_aug_acts = [f"Firstly, the person is {{}}", f"Secondly, the person is {{}}",
#                      f"Thirdly, the person is {{}}", f"Fourthly, the person is {{}}",
#                      f"Fifthly, the person is {{}}"]
#     text_no_acts = [f"The first action does not exist.",
#                     f"The second action does not exist.", f"The third action does not exist.",
#                     f"The fourth action does not exist.", f"The fifth action does not exist."]
#
#     text_dict_cnts = []
#     text_dict_acts = {}
#
#     for ii, txt in enumerate(text_aug_cnts):
#         text_dict_cnts.append(clip.tokenize(txt))
#     text_dict_cnts = torch.cat(text_dict_cnts)
#
#     for ii, txt in enumerate(text_aug_acts):
#         text_dict_acts[ii] = [clip.tokenize(txt.format(c)) for i, c in classes.items()]
#         text_dict_acts[ii].append(clip.tokenize(text_no_acts[ii]))
#         text_dict_acts[ii] = torch.cat(text_dict_acts[ii])
#     text_dict_acts = torch.cat([v for k, v in text_dict_acts.items()])
#
#     return text_dict_cnts, text_dict_acts


def text_prompt_slide_val_all(classes, cnt_max=5):
    if classes[0] == 'SIL':
        classes[0] = 'waiting and preparing'
        classes[48] = 'finishing and waiting'
    text_aug_cnts = [f"This clip contains no actions.",
                     f"This clip contains only one action.", f"This clip contains two actions.",
                     f"This clip contains three actions.", f"This clip contains four actions.",
                     f"This clip contains five actions.", f"This clip contains six actions."]
    text_aug_acts = [f"Firstly, ", f"Secondly, ",
                     f"Thirdly, ", f"Fourthly, ",
                     f"Fifthly, ", f"Sixthly, "]
    # text_aug_temp = [f"the person is {{}}.", f"the person is performing the action of {{}}.",
    #                  f"the character is {{}}.", f"he or she is {{}}.", f"the action {{}} is being played.",
    #                  f"it is the action of {{}}.", f"the human is {{}}.",
    #                  f"the person is working on {{}}.", f"the scene is {{}}.",
    #                  f"the person is focusing on {{}}.", f"the person is completing the action of {{}}.",
    #                  f"the step is {{}}", f"the action is {{}}.", f"the action step is {{}}."]
    text_aug_temp = [f"the person is {{}}.", f"the character is {{}}.", f"he or she is {{}}.",
                     f"the human is {{}}.", f"the scene is {{}}.", f"{{}} is being done.",
                     f"the step is {{}}", f"the action is {{}}.", f"the action step is {{}}."]
    text_no_acts = [f"The first action does not exist.",
                    f"The second action does not exist.", f"The third action does not exist.",
                    f"The fourth action does not exist.", f"The fifth action does not exist.",
                    f"The sixth action does not exist."]
    text_aug_cnts = text_aug_cnts[:cnt_max+1]
    text_aug_acts = text_aug_acts[:cnt_max]
    text_no_acts = text_no_acts[:cnt_max]

    num_temp = len(text_aug_temp)
    num_act = len(text_aug_acts)
    num_cnt = len(text_aug_cnts)
    res_token_cnt = []

    for id in range(num_cnt):
        res_token_cnt.append(clip.tokenize(text_aug_cnts[id]))
    res_token_cnt = torch.cat(res_token_cnt)

    res_token_acts = []

    for ii in range(num_act):
        res_token_acts.append([])
        for jj in range(num_temp):
            res_token_acts[ii].append([clip.tokenize(text_aug_acts[ii] + text_aug_temp[jj].format(c)) for i, c in
                                       classes.items()])
            res_token_acts[ii][jj].append(clip.tokenize(text_no_acts[ii]))
            res_token_acts[ii][jj] = torch.cat(res_token_acts[ii][jj])
        res_token_acts[ii] = torch.cat(res_token_acts[ii])
    res_token_acts = torch.cat(res_token_acts)
    # res_token_acts = res_token_acts.view(num_act, num_temp, len(classes)+1, -1)

    return res_token_cnt, res_token_acts, num_temp


# def text_prompt_slide(classes, id_list):
#     text_aug_1act = {1: f"The person is {{}}",
#                      2: f"The person is {{}} first, then {{}}",
#                      3: f"The person is {{}} first, then {{}} and {{}}",
#                      4: f"The person is {{}} first, then {{}}, then {{}} and {{}}",
#                      5: f"The person is {{}} first, then {{}}, then {{}}, then {{}} and {{}}"}
#     # num_text_aug = len(text_aug)
#
#     id_list_cnt = id_list >= 0
#     id_list_cnt = torch.sum(id_list_cnt, dim=1)
#     res_token = []
#
#     for ii, txt in enumerate(id_list):
#         action_list = []
#         for i in range(id_list_cnt[ii]):
#             action_list.append(classes[txt[i].item()])
#         if action_list[0] == 'SIL': action_list[0] = 'waiting and preparing'
#         if action_list[-1] == 'SIL': action_list[-1] = 'finishing and waiting'
#         sentence = text_aug_1act[id_list_cnt[ii].item()].format(*action_list)
#         token = clip.tokenize(sentence)
#         res_token.append(token)
#         # text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])
#     res_token = torch.cat(res_token)
#     # classes = torch.cat([v for k, v in text_dict.items()])
#
#     return res_token
#
#
# def text_prompt_slide_v2(classes, id_list, dataset):
#     text_aug_1act = {0: f"This clip contains no actions.",
#                      1: f"This clip contains only one action. The person is {{}}",
#                      2: f"This clip contains two actions. Firstly, the person is {{}}. Secondly, the person is {{}}",
#                      3: f"This clip contains three actions. Firstly, the person is {{}}. Secondly, the person is {{}}. "
#                         f"Thirdly, the person is {{}}",
#                      4: f"This clip contains four actions. Firstly, the person is {{}}. Secondly, the person is {{}}. "
#                         f"Thirdly, the person is {{}}. Fourthly, the person is {{}}",
#                      5: f"This clip contains five actions. Firstly, the person is {{}}. Secondly, the person is {{}}. "
#                         f"Thirdly, the person is {{}}. Fourthly, the person is {{}}. Fifthly, the person is {{}}"}
#     # num_text_aug = len(text_aug)
#
#     id_list_cnt = id_list >= 0
#     id_list_cnt = torch.sum(id_list_cnt, dim=1)
#     res_token = []
#
#     for ii, txt in enumerate(id_list):
#         action_list = []
#         for i in range(id_list_cnt[ii]):
#             action_list.append(classes[txt[i].item()])
#         if dataset == 'breakfast':
#             if action_list[0] == 'SIL': action_list[0] = 'waiting and preparing'
#             if action_list[-1] == 'SIL': action_list[-1] = 'finishing and waiting'
#         sentence = text_aug_1act[id_list_cnt[ii].item()].format(*action_list)
#         token = clip.tokenize(sentence)
#         res_token.append(token)
#         # text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])
#     res_token = torch.cat(res_token)
#     # classes = torch.cat([v for k, v in text_dict.items()])
#
#     return res_token
#
#
# def text_prompt_slide_v3(classes, id_list, dataset):
#     text_aug_cnts = [f"This clip contains no actions.",
#                      f"This clip contains only one action.", f"This clip contains two actions.",
#                      f"This clip contains three actions.", f"This clip contains four actions.",
#                      f"This clip contains five actions."]
#     text_aug_acts = [f"Firstly, the person is {{}}.", f"Secondly, the person is {{}}.",
#                      f"Thirdly, the person is {{}}.", f"Fourthly, the person is {{}}.",
#                      f"Fifthly, the person is {{}}."]
#     text_no_acts = [f"The first action does not exist.",
#                     f"The second action does not exist.", f"The third action does not exist.",
#                     f"The fourth action does not exist.", f"The fifth action does not exist."]
#
#     b, _ = id_list.shape
#     id_list_cnt = id_list >= 0
#     id_list_cnt = torch.sum(id_list_cnt, dim=1)
#     res_token_cnt = []
#
#     for id in id_list_cnt:
#         res_token_cnt.append(clip.tokenize(text_aug_cnts[id.item()]))
#     res_token_cnt = torch.cat(res_token_cnt)
#
#     res_token_acts = []
#     res_token_all = []
#
#     for ii, txt in enumerate(id_list):
#         num_acts = id_list_cnt[ii].item()
#         action_list = []
#         for i in range(num_acts):
#             action_list.append(classes[txt[i].item()])
#         if dataset == 'breakfast':
#             if action_list[0] == 'SIL': action_list[0] = 'waiting and preparing'
#             if action_list[-1] == 'SIL': action_list[-1] = 'finishing and waiting'
#         sentences = []
#         sentences_all = text_aug_cnts[num_acts]
#         for i in range(num_acts):
#             sent = text_aug_acts[i].format(action_list[i])
#             sentences.append(clip.tokenize(sent))
#             sentences_all += ' ' + sent
#         for i in range(num_acts, len(text_no_acts)):
#             sentences.append(clip.tokenize(text_no_acts[i]))
#         res_token_acts.append(torch.cat(sentences))
#         res_token_all.append(clip.tokenize(sentences_all))
#         # token = clip.tokenize(sentence)
#         # res_token.append(token)
#         # text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])
#     res_token_acts = torch.cat(res_token_acts).view(b, -1, res_token_cnt.shape[1])
#     res_token_all = torch.cat(res_token_all)
#     # classes = torch.cat([v for k, v in text_dict.items()])
#
#     return res_token_cnt, res_token_acts, res_token_all, id_list_cnt
#
#
# def text_prompt_slide_v4(classes, id_list, dataset, cnt_max=5):
#     text_aug_cnts = [f"This clip contains no actions.",
#                      f"This clip contains only one action,", f"This clip contains two actions,",
#                      f"This clip contains three actions,", f"This clip contains four actions,",
#                      f"This clip contains five actions,", f"This clip contains six actions,",
#                      f"This clip contains seven actions,", f"This clip contains eight actions,"]
#     text_aug_acts = [f"Firstly, ", f"Secondly, ", f"Thirdly, ", f"Fourthly, ",
#                      f"Fifthly, ", f"Sixthly, ", f"Seventhly, ", f"Eighthly, "]
#     text_aug_temp = [f"the person is {{}}.", f"the person is performing the action of {{}}.",
#                      f"the character is {{}}.", f"he or she is {{}}.", f"the action {{}} is being played.",
#                      f"it is the action of {{}}.", f"the human is {{}}.",
#                      f"the person is working on {{}}.", f"the scene is {{}}.",
#                      f"the person is focusing on {{}}.", f"the person is completing the action of {{}}.",
#                      f"the step is {{}}", f"the action is {{}}.", f"the action step is {{}}."]
#     text_long_temp = [f"the person is {{}}.", f"the character is {{}}.", f"he or she is {{}}.",
#                       f"the human is {{}}.", f"the scene is {{}}.", f"{{}} is being done.",
#                       f"the step is {{}}", f"the action is {{}}.", f"the action step is {{}}."]
#     text_no_acts = [f"The first action does not exist.",
#                     f"The second action does not exist.", f"The third action does not exist.",
#                     f"The fourth action does not exist.", f"The fifth action does not exist.",
#                     f"The sixth action does not exist.", f"The seventh action does not exist.",
#                     f"The eighth action does not exist."]
#     text_aug_cnts = text_aug_cnts[:cnt_max+1]
#     text_aug_acts = text_aug_acts[:cnt_max]
#     text_no_acts = text_no_acts[:cnt_max]
#
#     b, _ = id_list.shape
#     num_temp = len(text_aug_temp)
#     num_long = len(text_long_temp)
#     text_id = np.random.randint(num_temp, size=len(id_list) * cnt_max).reshape(-1, cnt_max)
#     text_id_long = np.random.randint(num_long, size=len(id_list) * cnt_max).reshape(-1, cnt_max)
#     id_list_cnt = id_list >= 0
#     id_list_cnt = torch.sum(id_list_cnt, dim=1)
#     res_token_cnt = []
#
#     for id in id_list_cnt:
#         res_token_cnt.append(clip.tokenize(text_aug_cnts[id.item()]))
#     res_token_cnt = torch.cat(res_token_cnt)
#
#     res_token_acts = []
#     res_token_all = []
#
#     for ii, txt in enumerate(id_list):
#         num_acts = id_list_cnt[ii].item()
#         action_list = []
#         for i in range(num_acts):
#             action_list.append(classes[txt[i].item()])
#         if dataset == 'breakfast':
#             if action_list[0] == 'SIL': action_list[0] = 'waiting and preparing'
#             if action_list[-1] == 'SIL': action_list[-1] = 'finishing and waiting'
#         sentences = []
#         sentences_all = text_aug_cnts[num_acts]
#         for i in range(num_acts):
#             sent = text_aug_acts[i] + text_aug_temp[text_id[ii][i]].format(action_list[i])
#             sentences.append(clip.tokenize(sent))
#             sentences_all += ' ' + text_aug_acts[i] + text_long_temp[text_id_long[ii][i]].format(action_list[i])
#         for i in range(num_acts, len(text_no_acts)):
#             sentences.append(clip.tokenize(text_no_acts[i]))
#         res_token_acts.append(torch.cat(sentences))
#         res_token_all.append(clip.tokenize(sentences_all))
#         # token = clip.tokenize(sentence)
#         # res_token.append(token)
#         # text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])
#     res_token_acts = torch.cat(res_token_acts).view(b, -1, res_token_cnt.shape[1])
#     res_token_all = torch.cat(res_token_all)
#     # classes = torch.cat([v for k, v in text_dict.items()])
#
#     return res_token_cnt, res_token_acts, res_token_all, id_list_cnt


if __name__ == '__main__':
    cls = {0: 'background', 1: 'closing ketchup', 2: 'closing jam', 3: 'putting chocolate', 4: 'opening chocolate',
           5: 'opening tea', 6: 'putting tea', 7: 'pouring sugar into the cup with a spoon', 8: 'putting peanut',
           9: 'taking water', 10: 'stirring in the cup with a spoon', 11: 'closing chocolate', 12: 'taking honey',
           13: 'scooping coffee with a spoon', 14: 'taking peanut', 15: 'putting ketchup', 16: 'closing mayonnaise',
           17: 'folding bread', 18: 'opening ketchup', 19: 'putting bread on cheese and bread', 20: 'opening water',
           21: 'taking bread', 22: 'closing honey', 23: 'taking mustard', 24: 'putting mustard',
           25: 'scooping peanut with a spoon', 26: 'pouring water into the cup', 27: 'scooping jam with a spoon',
           28: 'stirring in the cup', 29: 'spreading jam on bread with a spoon',
           30: 'pouring coffee into the cup with a spoon', 31: 'putting honey', 32: 'opening peanut', 33: 'taking sugar',
           34: 'opening mayonnaise', 35: 'pouring mustard on hotdog and bread', 36: 'taking mayonnaise',
           37: 'pouring honey on bread', 38: 'putting water', 39: 'taking coffee',
           40: 'pouring ketchup on hotdog and bread', 41: 'pouring mayonnaise on cheese and bread',
           42: 'pouring chocolate on bread', 43: 'putting cheese on bread', 44: 'opening honey', 45: 'closing sugar',
           46: 'putting hotdog on bread', 47: 'pouring mustard on cheese and bread', 48: 'opening jam',
           49: 'opening cheese', 50: 'scooping sugar with a spoon', 51: 'spreading peanut on bread with a spoon',
           52: 'taking spoon', 53: 'putting bread on bread', 54: 'taking cheese', 55: 'putting sugar', 56: 'opening sugar',
           57: 'opening coffee', 58: 'opening mustard', 59: 'putting jam', 60: 'closing peanut', 61: 'taking tea',
           62: 'closing mustard', 63: 'closing water', 64: 'putting coffee', 65: 'taking cup', 66: 'taking jam',
           67: 'shaking tea in the cup', 68: 'closing coffee', 69: 'pouring honey into the cup', 70: 'taking ketchup',
           71: 'putting mayonnaise', 72: 'taking chocolate', 73: 'taking hotdog'}
    cls = {int(k): v for k, v in cls.items()}
    id_list = torch.tensor([[30, 31, 32, -1, -1, -1, -1, -1, -1, -1],
                            [52, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [26, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [29, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [68, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [65, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [13, 30, -1, -1, -1, -1, -1, -1, -1, -1],
                            [32, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [67, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [41, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [61, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [36, 34, -1, -1, -1, -1, -1, -1, -1, -1],
                            [65, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [21, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    # text_prompt_slide_allwocnt(cls, id_list, 'gtea')
    text_prompt_slide_val_all(cls)
