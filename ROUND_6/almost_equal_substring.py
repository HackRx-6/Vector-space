def smallest_almost_equal_index(s, pattern):
    m, n = len(s), len(pattern)
    for start in range(m - n + 1):
        sub = s[start:start+n]
        diff_count = sum(1 for a, b in zip(sub, pattern) if a != b)
        if diff_count <= 1:
            return start
    return -1

# Input
s = "ababbababa"
pattern = "bacaba"

# Output the answer
result = smallest_almost_equal_index(s, pattern)
print(result)
