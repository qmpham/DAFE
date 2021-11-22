

def decode(s, bi_dict):
    #print(s)
    ans = []
    l = max([len(k) for k in bi_dict.keys()])
    if s=="":
        return [""]
    for i in range(l):
        if s[:i+1] in bi_dict:
            #print(s[:i+1])
            next_ans = decode(s[i+1:],bi_dict)
            #print(next_ans)
            for temp_ans in next_ans:
                #print(temp_ans)
                ans.append(bi_dict[s[:i+1]]+temp_ans)
            #print(ans)
    
    return ans

bi_dict_ = {'01':'e','10':'f'}
s = '01101001'
print(decode(s,bi_dict_))