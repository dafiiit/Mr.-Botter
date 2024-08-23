import math

# Constants
TOPS = 13 * 10**12  # 13 Tera Operations per Second
RAM_LIMIT_BYTES = 6 * 10**9  # 6 GB in Bytes
FLOAT_SIZE_BYTES = 4  # 32-bit float size in bytes

# Function to calculate required RAM for given network configuration
def calculate_ram(layers):
    total_weights = sum([layers[i] * layers[i + 1] for i in range(len(layers) - 1)])
    total_biases = sum(layers)
    total_ram = (total_weights + total_biases) * FLOAT_SIZE_BYTES
    return total_ram


# Function to calculate operations required for given network configuration
def calculate_operations(layers):
    total_operations = 0
    for i in range(len(layers) - 1):
        n = layers[i]
        next_n = layers[i + 1]
        # Using the formula: 2n^3 - n^2 for each layer transition
        total_operations += 2 * (n**3) - (n**2)
    return total_operations


# Function to calculate maximum neurons per layer based on TOPS and RAM constraints
def calculate_layers(num_layers, MULTIPLIKATION_PER_SEC):
    # Start with a small number of neurons
    n = 10
    layers = [n**2] * num_layers

    while True:
        total_operations = calculate_operations(layers)
        total_ram = calculate_ram(layers)

        if total_operations * MULTIPLIKATION_PER_SEC < TOPS and total_ram < RAM_LIMIT_BYTES:
            # Try increasing the neurons per layer
            n += 1
            layers = [n**2] * num_layers
        else:
            # If we exceed either limit, backtrack slightly to ensure we stay within bounds
            n -= 1
            layers = [n**2] * num_layers
            break

    return layers


# Main program
def main():
    num_layers = int(input("Enter the number of layers: "))
    MULTIPLIKATION_PER_SEC = int(input("Enter the number of Multiplications per Second: "))
    layers = calculate_layers(num_layers, MULTIPLIKATION_PER_SEC)

    print("\nCalculated Layer Sizes:")
    for i, neurons in enumerate(layers):
        print(f"Layer {i+1}: {neurons} neurons")

    total_ram = calculate_ram(layers)
    print(f"\nTotal estimated RAM usage: {total_ram / (1024**3):.2f} GB")

    total_operations = calculate_operations(layers)
    print(f"Total operations per second: {total_operations * 10:.2e}")


if __name__ == "__main__":
    main()
