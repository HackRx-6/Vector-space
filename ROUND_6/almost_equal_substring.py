def almost_equal_substring(s, pattern):
    n, m = len(s), len(pattern)
    for i in range(n - m + 1):
        substring = s[i:i + m]
        diff_count = 0
        for a, b in zip(substring, pattern):
            if a != b:
                diff_count += 1
                if diff_count > 1:
                    break
        if diff_count <= 1:
            return i
    return -1

# Input values
s = "abcdefg"
pattern = "bcdffg"

result = almost_equal_substring(s, pattern)
print(result)
