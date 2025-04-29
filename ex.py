# arr = [1, 2, 2, 3, 2, 1]
#
# dic = {}
# for i in arr:
#     dic[i] = dic.get(i, 0) + 1
#
# print(len(arr) - max(dic.values()))


# arr = [1,2]
#
# for i in range(len(arr)):
#     for j in range(len(arr)):
#         print(arr[i],arr[j])
#
# #
# arr = "123695"
# n = list(arr)
# n.sort(reverse=True)
# print(int("".join(n)))
#

# a = "qwwweeessdrrfvvggyyhuujb"
#
# for i in a:
#     if a.count(i) == 1:
#         print(i)
#         break

import math
a = 1
for i in range(a):
    x,y,z = list(map(int,input().split()))
    n = math.log(x,2)
    print(int((n*y)+((n-1)*z)))