def find_almost_equal_index(s, pattern):
    n = len(s)
    m = len(pattern)

    for i in range(n - m + 1):
        sub = s[i:i + m]
        diff = sum(1 for a, b in zip(sub, pattern) if a != b)
        if diff <= 1:
            return i
    return -1

# Test the function with provided input
s = "abcdefg"
pattern = "bcdffg"
result = find_almost_equal_index(s, pattern)
print(result)
