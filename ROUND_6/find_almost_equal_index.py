def find_almost_equal_index(s, pattern):
    n, m = len(s), len(pattern)
    for i in range(n - m + 1):
        mismatch = 0
        for j in range(m):
            if s[i + j] != pattern[j]:
                mismatch += 1
                if mismatch > 1:
                    break
        if mismatch <= 1:
            return i
    return -1

# Test the function based on the problem statement
s = "ababbababa"
pattern = "bacaba"
output = find_almost_equal_index(s, pattern)
print(output)
