```python
def fibonacci(n):
    """
    Generate the Fibonacci sequence up to the nth number.

    Args:
        n (int): The number of Fibonacci numbers to generate.

    Returns:
        list: A list of Fibonacci numbers.
    """
    # Initialize the Fibonacci sequence with the first two numbers
    fib_sequence = [0, 1]

    # Generate the Fibonacci sequence up to the nth number
    while len(fib_sequence) < n:
        # Calculate the next Fibonacci number as the sum of the last two numbers
        next_number = fib_sequence[-1] + fib_sequence[-2]
        # Append the next number to the Fibonacci sequence
        fib_sequence.append(next_number)

    return fib_sequence


def fibonacci_recursive(n):
    """
    Generate the nth Fibonacci number using a recursive approach.

    Args:
        n (int): The position of the Fibonacci number to generate.

    Returns:
        int: The nth Fibonacci number.
    """
    # Base cases: F(0) = 0, F(1) = 1
    if n == 0:
        return 0
    elif n == 1:
        return 1
    # Recursive case: F(n) = F(n-1) + F(n-2)
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)


def main():
    # Example usage: Generate the first 10 Fibonacci numbers
    n = 10
    print("Fibonacci sequence (iterative):", fibonacci(n))

    # Example usage: Generate the 10th Fibonacci number using recursion
    print("10th Fibonacci number (recursive):", fibonacci_recursive(n))


if __name__ == "__main__":
    main()
```