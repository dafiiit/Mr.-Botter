# Constants
TOPS = 13 * 10**12  # 13 Tera Operations per Second
RAM_LIMIT_BYTES = 6 * 10**9  # 6 GB in Bytes
FLOAT_SIZE_BYTES = 4  # 32-bit float size in bytes


# Function to calculate required RAM for transformer
def calculate_ram_transformer(layers, d_model, num_heads):
    total_weights = sum(
        [d_model**2 * 3 * (len(layers) - 1)]
    )  # Simplified for layers and self-attention
    total_biases = sum([d_model * 2 * len(layers)])
    total_ram = (total_weights + total_biases) * FLOAT_SIZE_BYTES
    return total_ram


# Function to calculate operations required for transformer
def calculate_operations_transformer(layers, d_model, num_heads):
    total_operations = 0
    for i in range(len(layers)):
        n = d_model
        # Operations for Self-Attention
        total_operations += n**2 * n * num_heads  # Simplified for self-attention
        total_operations += 2 * n**2 * n  # Simplified for feedforward
    return total_operations


# Function to calculate maximum neurons per layer based on TOPS and RAM constraints for transformer
def calculate_layers_transformer(
    num_layers, d_model, num_heads, MULTIPLIKATION_PER_SEC
):
    # Start with a small number of neurons
    n = 64  # Start with a reasonable size
    layers = [n] * num_layers

    while True:
        total_operations = calculate_operations_transformer(layers, d_model, num_heads)
        total_ram = calculate_ram_transformer(layers, d_model, num_heads)

        if (
            total_operations * MULTIPLIKATION_PER_SEC < TOPS
            and total_ram < RAM_LIMIT_BYTES
        ):
            # Try increasing the neurons per layer
            n += 16
            layers = [n] * num_layers
        else:
            # If we exceed either limit, backtrack slightly to ensure we stay within bounds
            n -= 16
            layers = [n] * num_layers
            break

    return layers


# Main program for Transformer
def main_transformer():
    num_layers = int(input("Enter the number of layers: "))
    d_model = int(input("Enter the model dimension: "))
    num_heads = int(input("Enter the number of attention heads: "))
    MULTIPLIKATION_PER_SEC = int(
        input("Enter the number of Multiplications per Second: ")
    )
    layers = calculate_layers_transformer(
        num_layers, d_model, num_heads, MULTIPLIKATION_PER_SEC
    )

    print("\nCalculated Layer Sizes:")
    for i, neurons in enumerate(layers):
        print(f"Layer {i+1}: {neurons} neurons")

    total_ram = calculate_ram_transformer(layers, d_model, num_heads)
    print(f"\nTotal estimated RAM usage: {total_ram / (1024**3):.2f} GB")

    total_operations = calculate_operations_transformer(layers, d_model, num_heads)
    print(f"Total operations per second: {total_operations * 10:.2e}")


if __name__ == "__main__":
    main_transformer()
