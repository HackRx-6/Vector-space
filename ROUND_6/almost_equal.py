def smallest_almost_equal_index(s, pattern):
    n = len(s)
    m = len(pattern)
    for i in range(n - m + 1):
        sub = s[i:i + m]
        diff = sum(sub[j] != pattern[j] for j in range(m))
        if diff <= 1:
            return i
    return -1

# Input strings
s = "abcdefg"
pattern = "bcdffg"

# Output result
result = smallest_almost_equal_index(s, pattern)
print(result)
