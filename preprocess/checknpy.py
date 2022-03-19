import numpy as np

top1 = 0
top5 = 0
top1_wc = 0
top5_wc = 0
top1_cnt = 0
num_act = 0
num_cnt = 0
dataset = 'gtea2salad'
for n_split in range(5, 6):
    n_split = str(n_split)
    corr_numact = [0, 0, 0, 0, 0, 0, 0]
    num_numact = [0, 0, 0, 0, 0, 0, 0]
    corr_1 = 0
    corr_5 = 0
    corr_1_wcnt = 0
    corr_5_wcnt = 0
    corr_1_cnt = 0
    final_act_1 = np.load("./prompt_test/"+dataset+"/split"+n_split+"/final_act_1.npy")
    final_act_5 = np.load("./prompt_test/"+dataset+"/split"+n_split+"/final_act_5.npy")
    final_cnt = np.load("./prompt_test/"+dataset+"/split"+n_split+"/final_cnt_1.npy")
    gt_act = np.load("./prompt_test/"+dataset+"/split"+n_split+"/gt_act.npy")
    gt_cnt = gt_act >= 0
    gt_cnt = np.sum(gt_cnt, axis=1)
    num_cnt += len(gt_cnt)
    for i in range(len(gt_cnt)):
        num_act += gt_cnt[i]
        num_numact[gt_cnt[i]] += gt_cnt[i]
        corr_now = 0
        for k in range(gt_cnt[i]):
            if final_act_1[i][k] == gt_act[i][k]:
                corr_1 += 1
                corr_now += 1
            if gt_act[i][k] in final_act_5[i][k]:
                corr_5 += 1
        corr_numact[gt_cnt[i]] += corr_now
        for k in range(final_cnt[i][0]):
            if final_act_1[i][k] == gt_act[i][k]:
                corr_1_wcnt += 1
            if gt_act[i][k] in final_act_5[i][k]:
                corr_5_wcnt += 1
    for i in range(len(gt_cnt)):
        if final_cnt[i][0] == gt_cnt[i]:
            corr_1_cnt += 1
    top1 += float(corr_1)
    top5 += float(corr_5)
    top1_wc += float(corr_1_wcnt)
    top5_wc += float(corr_5_wcnt)
    top1_cnt += float(corr_1_cnt)
    num_numact = [i if i != 0 else 1 for i in num_numact]
    top1_numact = [float(corr_numact[i]) / num_numact[i] * 100 for i in range(1, len(num_numact))]

top1 /= num_act
top5 /= num_act
top1_wc /= num_act
top5_wc /= num_act
top1_cnt /= num_cnt
print('Top1: {}/{}, Top5: {}/{}, Top1_cnt: {}'.format(top1, top1_wc, top5, top5_wc, top1_cnt))
