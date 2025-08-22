def find_almost_equal_index(s: str, pattern: str) -> int:
    n = len(s)
    m = len(pattern)

    for i in range(n - m + 1):
        substring = s[i:i + m]
        difference_count = sum(1 for a, b in zip(substring, pattern) if a != b)
        if difference_count <= 1:
            return i
    return -1

s = "ababbababa"
pattern = "bacaba"
result = find_almost_equal_index(s, pattern)
print(result)
