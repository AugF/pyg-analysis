labels = ['layer0', 'other']
for label in labels:
    if label == 'other':
        # log_soft_max: start_time,  eval end time
        sql = "select start, end, text from nvtx_events where text like '{}'"
        log_res = cur.execute(sql.format('log_softmax%')).fetchall() # log_softmax
        forward_res = cur.execute(sql.format('forward')).fetchall()[1:]
        backward_res = cur.execute(sql.format('backward')).fetchall()[1:]
        eval_res = cur.execute(sql.format('eval')).fetchall()
        other_time = 0
        for i in range(50):
            if i in outliers: continue
            a1 = get_real_time(forward_res[i], cur)
            a2 = get_real_time(log_res[2 * i], cur)
            t1 = (a1[2] - a2[1]) / 1e6 # forward
            t2 = (get_real_time(eval_res[i], cur)[2] - get_real_time(log_res[2 * i + 1], cur)[1]) / 1e6
            other_time += t1 + t2
            # 计算others在backward所对应的时间
            id = get_int(log_res[2 * i][2]) # 获取id
            seq_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {}%'"
            btime = cur.execute(seq_sql.format(id)).fetchone()
            max_time = get_real_time(btime, cur) # 寻找结束时间
            # 开始时间 = backward的开始时间
            min_time = get_real_time(backward_res[i], cur)
            print(max_time, min_time)
            other_time += (max_time[2] - min_time[1]) / 1e6 # 注意检查每一个都要除以1e6
            break
        other_time /= (50 - len(outliers)) # 求出的是每个epoch的平均值
    else:
        sql = "select start, end, text from nvtx_events where text == '{}'".format(label)
        res = cur.execute(sql).fetchall()[1:]  # 不考虑warm up
        cost_time = 0
        for i in range(50):
            if i in outliers: continue  # 过滤掉异常的情况
            # 2*i: forward; 2*i+1: eval
            cost_time += get_real_time(res[2 * i + 1], cur)[0]  # eval_time
            ltime, rtime = res[2 * i][0], res[2 * i][1]
            cost_time += get_real_time(res[2 * i], cur)[0]  # forward_time

            # backward_time, 只对forward阶段进行思考
            backward_time = 0
            # 1. 找当前包含的nvtx_times的seq
            seq_texts = cur.execute(
                "select text from nvtx_events where start >= {} and end <= {} and text like '%seq%'".format(ltime, rtime)).fetchall()
            min_seq, max_seq = get_int(seq_texts[0][0]), get_int(seq_texts[-1][0])
            seq_sql = "select start, end, text from nvtx_events where text like '%Backward%seq = {}%'"
            btime = cur.execute(seq_sql.format(max_seq)).fetchone()
            min_time = get_real_time(btime, cur)[1]
            btime = cur.execute(seq_sql.format(min_seq)).fetchone()
            max_time = get_real_time(btime, cur)[2]
            cost_time += (max_time - min_time) / 1e6
        cost_time /= (50 - len(outliers))

