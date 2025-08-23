def almost_equal_substring(s, pattern):
    n, m = len(s), len(pattern)
    for i in range(n - m + 1):
        diff = 0
        for j in range(m):
            if s[i + j] != pattern[j]:
                diff += 1
            if diff > 1:
                break
        if diff <= 1:
            return i
    return -1

# Given test case
s = "abcdefg"
pattern = "bcdffg"
result = almost_equal_substring(s, pattern)
print(result)
