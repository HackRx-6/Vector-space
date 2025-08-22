def smallest_almost_equal_index(s, pattern):
    n, m = len(s), len(pattern)
    for i in range(n - m + 1):
        sub = s[i:i+m]
        diff = sum(1 for a, b in zip(sub, pattern) if a != b)
        if diff <= 1:
            return i
    return -1

# Example usage:
s = "abcdefg"
pattern = "bcdffg"
result = smallest_almost_equal_index(s, pattern)
print(result)  # Expected output: 1
