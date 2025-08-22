import sys
from almost_equal import find_almost_equal_index

s = "abcdefg"
pattern = "bcdffg"
answer = find_almost_equal_index(s, pattern)
with open("ROUND_6/output.txt", "w") as f:
    f.write(str(answer))
print(answer)
