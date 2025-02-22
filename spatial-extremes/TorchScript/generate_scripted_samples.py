import torch

def main():
    # Load the TorchScript model saved as 'scripted_GANGeneratorWrapper.pt'
    scripted_model = torch.jit.load(r"C:\Users\reverd\Repositories\master-thesis\spatial-extremes\TorchScript\scripted_GANGenerator.pt")
    print("Scripted model loaded successfully.")

    # Define the noise dimension (should match the one used when scripting)
    noise_dim = 100  # Change this if your model uses a different noise dimension
    
    # Create a dummy noise tensor with batch size 4 (for example)
    test_noise = torch.randn(4, noise_dim)
    
    # Use the exported generate_samples function to produce outputs
    generated_samples = scripted_model.generate_samples(test_noise)
    
    # Print the shape of the generated samples
    print("Generated samples shape:", generated_samples.shape)
    
    # Optionally, inspect a few values from the output tensor
    print("Sample output (first generated sample):", generated_samples[0])

if __name__ == "__main__":
    main()