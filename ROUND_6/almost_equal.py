def find_almost_equal_index(s, pattern):
    n, m = len(s), len(pattern)
    for i in range(n - m + 1):
        substring = s[i:i+m]
        diff = sum(1 for a, b in zip(substring, pattern) if a != b)
        if diff <= 1:
            return i
    return -1

if __name__ == "__main__":
    s = "abcdefg"
    pattern = "bcdffg"
    answer = find_almost_equal_index(s, pattern)
    print(answer)
