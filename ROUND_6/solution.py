def find_almost_equal_substring(s, pattern):
    length = len(pattern)
    for i in range(len(s) - length + 1):
        diffs = sum(1 for j in range(length) if s[i+j] != pattern[j])
        if diffs <= 1:
            return i
    return -1

# Test call
s = "ababbababa"
pattern = "bacaba"
result = find_almost_equal_substring(s, pattern)
print(result)
